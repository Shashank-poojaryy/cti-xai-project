DEMO IMAGES for the XAI Reliability Streamlit app
==================================================
All images are from the HELD-OUT TEST split (patient-independent; the models
never saw these patients during training).

cardiomegaly/  27 confirmed Cardiomegaly images, each with a ground-truth
               bounding box. Files are ordered by mean CTI (see manifest.csv);
               the top ones give the cleanest heatmaps over the heart.
normal/        15 confirmed "No Finding" (Normal) images.

HOW TO USE
1. In the app, upload any file from cardiomegaly/ (start with the highest-CTI ones).
2. Select a model (DenseNet121 = most trustworthy) and an XAI method
   (Grad-CAM++ ranked #1). Click Analyze.
3. The heatmap should highlight the cardiac silhouette for Cardiomegaly cases.

NOTE ON PREDICTIONS
The models are tuned for high sensitivity (they rarely miss cardiomegaly), so at
the default 0.5 threshold they OVER-predict positive: some Normal images may be
labelled "Cardiomegaly" with low confidence. That is expected from the class-
imbalance handling -- judge the heatmap quality, not just the label.

manifest.csv lists every demo image with its true label, patient id, and CTI.
