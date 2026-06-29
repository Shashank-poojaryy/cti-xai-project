"""
Step 6 - additional Q1/Q2 analyses on the per-image CTI table.

Produces:
  results/pathology_correlation.csv   bbox area vs CTI (Spearman r, p, 95% CI) per method
  results/sensitivity_analysis.csv    CTI + rank under 3 weight configs
  results/statistical_tests.csv       Friedman, Wilcoxon (Bonferroni), Cohen's d
  results/component_rankings.csv      per-component method ranking

CTI per (image, method) is averaged across the 3 models -> 146 paired samples/method.
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon

DATA_DIR    = r"C:\Users\NMAMIT\cti_project\data"
RESULTS_DIR = r"C:\Users\NMAMIT\cti_project\results"
PER_IMAGE   = os.path.join(RESULTS_DIR, "cti_per_image.csv")
METHODS     = ["gradcam", "gradcampp", "scorecam", "layercam", "integrated_gradients"]
COMPONENTS  = ["localization", "stability", "cross_xai", "cross_arch", "sanity"]


def per_image_method(df):
    """CTI per (image, method) averaged across models. Returns wide table:
    index=image_id, columns=method, values=cti."""
    g = df.groupby(["image_id", "method"])["cti"].mean().reset_index()
    return g.pivot(index="image_id", columns="method", values="cti")


def pathology_correlation(df):
    bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    bb = bb[bb["Finding Label"] == "Cardiomegaly"].drop_duplicates("Image Index")
    s2 = (224 / 1024) ** 2
    bb["area"] = (bb["w"] * bb["h]"]).astype(float) * s2
    bb["image_id"] = bb["Image Index"].str.replace(".png", "", regex=False)
    area = bb.set_index("image_id")["area"]

    rows = []
    for m in METHODS:
        sub = df[df["method"] == m].groupby("image_id")["cti"].mean()
        common = sub.index.intersection(area.index)
        a, c = area.loc[common].values, sub.loc[common].values
        r, p = spearmanr(a, c)
        # bootstrap 95% CI for r
        rng = np.random.default_rng(42); boot = []
        for _ in range(1000):
            idx = rng.choice(len(a), len(a), replace=True)
            if len(np.unique(a[idx])) > 1:
                boot.append(spearmanr(a[idx], c[idx])[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"method": m, "n": len(common), "spearman_r": round(r, 4),
                     "p_value": round(p, 5), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "significant": p < 0.05})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "pathology_correlation.csv"), index=False)
    print("\nPathology size vs CTI:"); print(out.to_string(index=False))
    return out


def sensitivity(df):
    configs = {"equal": [.2, .2, .2, .2, .2],
               "loc_heavy": [.30, .20, .15, .20, .15],
               "stab_heavy": [.15, .30, .20, .20, .15]}
    g = df.groupby("method")[COMPONENTS].mean()
    rows = []
    for cfg, w in configs.items():
        cti = (g.values * np.array(w)).sum(axis=1)
        s = pd.Series(cti, index=g.index)
        ranks = s.rank(ascending=False).astype(int)
        for m in g.index:
            rows.append({"config": cfg, "method": m,
                         "cti": round(s[m], 4), "rank": int(ranks[m])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "sensitivity_analysis.csv"), index=False)
    piv = out.pivot(index="method", columns="config", values="rank")
    print("\nSensitivity (rank per config):"); print(piv.to_string())
    stable = (piv.nunique(axis=1) == 1).all()
    print(f"Ranking stable across all configs: {stable}")
    return out


def stat_tests(df):
    wide = per_image_method(df).dropna()
    rows = []
    # Friedman across 5 methods
    chi, p = friedmanchisquare(*[wide[m].values for m in METHODS])
    rows.append({"test": "Friedman", "comparison": "all 5 methods",
                 "statistic": round(chi, 4), "p_value": f"{p:.3e}",
                 "df": len(METHODS) - 1, "significant": p < 0.05})
    # Wilcoxon pairwise + Bonferroni
    alpha = 0.05 / 10
    for a, b in itertools.combinations(METHODS, 2):
        try:
            w, pw = wilcoxon(wide[a].values, wide[b].values)
        except ValueError:
            w, pw = np.nan, 1.0
        rows.append({"test": "Wilcoxon", "comparison": f"{a} vs {b}",
                     "statistic": round(w, 4) if not np.isnan(w) else "NA",
                     "p_value": f"{pw:.3e}", "df": "",
                     "significant": pw < alpha})
    # Cohen's d best vs worst
    means = wide.mean()
    best, worst = means.idxmax(), means.idxmin()
    x, y = wide[best].values, wide[worst].values
    sp = np.sqrt(((len(x)-1)*x.std(ddof=1)**2 + (len(y)-1)*y.std(ddof=1)**2) / (len(x)+len(y)-2))
    d = (x.mean() - y.mean()) / sp if sp else 0.0
    rows.append({"test": "Cohen_d", "comparison": f"{best} vs {worst}",
                 "statistic": round(d, 4), "p_value": "", "df": "", "significant": ""})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False)
    print("\nStatistical tests:"); print(out.to_string(index=False))
    print(f"Bonferroni alpha = {alpha}")
    return out


def component_rankings(df):
    g = df.groupby("method")[COMPONENTS].mean()
    rows = []
    for c in COMPONENTS:
        ranks = g[c].rank(ascending=False).astype(int)
        winner = g[c].idxmax()
        for m in g.index:
            rows.append({"component": c, "method": m,
                         "score": round(g.loc[m, c], 4), "rank": int(ranks[m])})
        print(f"  {c:14s} winner: {winner} ({g[c].max():.4f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "component_rankings.csv"), index=False)
    return out


def cti_confidence(df):
    """Bootstrap 95% CI (over images) for each method's mean CTI."""
    wide = per_image_method(df)
    rng = np.random.default_rng(42)
    rows = []
    for m in METHODS:
        vals = wide[m].dropna().values
        boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"method": m, "mean_cti": round(vals.mean(), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(vals)})
    out = pd.DataFrame(rows).sort_values("mean_cti", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    out.to_csv(os.path.join(RESULTS_DIR, "cti_confidence_intervals.csv"), index=False)
    print("\nCTI 95% CIs (bootstrap, n=2000):"); print(out.to_string(index=False))
    return out


def testsplit_ranking(df):
    """Leakage-free robustness check: CTI ranking on bbox images that fall in
    the held-out TEST split only (zero patient overlap with training)."""
    test_ids = set(pd.read_csv(os.path.join(DATA_DIR, "test.csv"))["Image Index"]
                   .str.replace(".png", "", regex=False))
    sub = df[df["image_id"].isin(test_ids)]
    n_imgs = sub["image_id"].nunique()
    rank = sub.groupby("method")["cti"].mean().sort_values(ascending=False)
    out = rank.reset_index(); out.columns = ["method", "mean_cti"]
    out["mean_cti"] = out["mean_cti"].round(4); out["rank"] = range(1, len(out) + 1)
    out.to_csv(os.path.join(RESULTS_DIR, "cti_results_testsplit.csv"), index=False)
    print(f"\nLeakage-free ranking (test-split bbox images only, n={n_imgs}):")
    print(out.to_string(index=False))
    return out


def main():
    df = pd.read_csv(PER_IMAGE)
    print(f"Loaded {len(df)} per-image rows")
    pathology_correlation(df)
    sensitivity(df)
    stat_tests(df)
    print("\nPer-component winners:")
    component_rankings(df)
    cti_confidence(df)
    testsplit_ranking(df)
    print("\nStep 6 analysis complete.")


if __name__ == "__main__":
    main()
