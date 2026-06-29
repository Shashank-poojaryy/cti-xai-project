"""
One-time preprocessing cache: decode + resize every split image to 224x224
uint8 and store as a memory-mapped array on disk. Training then reads
pre-resized data (instant) instead of decoding 1024px PNGs every epoch,
turning the pipeline from CPU-decode-bound (~46 img/s) to GPU-bound.

Uses a ThreadPoolExecutor (PIL releases the GIL during decode/resize) -> safe
on Windows, unlike DataLoader multiprocessing (num_workers>0).

Outputs in data/cache/:
  {split}_imgs.dat    uint8 memmap, shape (N, 224, 224, 3)
  {split}_labels.npy  int8,  shape (N,)
  {split}_names.npy   image filenames in row order
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

DATA_DIR  = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR = r"C:\Users\NMAMIT\cti_project\images"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SIZE = 224
os.makedirs(CACHE_DIR, exist_ok=True)


def build_split(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    names = df["Image Index"].values
    labels = df["label"].values.astype(np.int8)
    n = len(df)
    img_path = os.path.join(CACHE_DIR, f"{split}_imgs.dat")

    if os.path.exists(img_path) and os.path.getsize(img_path) == n * SIZE * SIZE * 3:
        print(f"[{split}] cache already present ({n} imgs) - skipping")
        return

    arr = np.memmap(img_path, dtype=np.uint8, mode="w+", shape=(n, SIZE, SIZE, 3))

    def load_one(i):
        im = Image.open(os.path.join(IMAGE_DIR, names[i])).convert("RGB").resize((SIZE, SIZE))
        arr[i] = np.asarray(im, dtype=np.uint8)

    t0 = time.time()
    done = [0]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for _ in ex.map(load_one, range(n)):
            done[0] += 1
            if done[0] % 5000 == 0:
                print(f"  [{split}] {done[0]}/{n}  ({done[0]/(time.time()-t0):.0f} img/s)")
    arr.flush()
    np.save(os.path.join(CACHE_DIR, f"{split}_labels.npy"), labels)
    np.save(os.path.join(CACHE_DIR, f"{split}_names.npy"), names)
    print(f"[{split}] cached {n} imgs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        build_split(split)
    print("Cache build complete.")
