import csv
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

rows = [r for r in csv.DictReader(open("out/sweep.csv", encoding="utf-8"))
        if r["status"] == "ok"]


def g(r, k):
    try:
        return float(r[k])
    except Exception:
        return np.nan


def auc(pos, neg):
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size < 3 or neg.size < 3:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


pos = [r for r in rows if r["label"] == "confirmed"]
neg = [r for r in rows if r["label"] == "fp"]
print("AUC (0.5 = useless, 1.0 = perfect; oriented so higher = more planet-like)\n")

tests = {
    "depth/koi_depth (prior)": lambda r: g(r, "depth_ppm") / g(r, "koi_depth_ppm"),
    "period match (prior)": lambda r: -abs(g(r, "period") - g(r, "koi_period")) / g(r, "koi_period"),
    "snr_power": lambda r: g(r, "snr_power"),
    "1/odd_even_ratio": lambda r: 1.0 / max(g(r, "odd_even_ratio"), 1e-6),
    "1/centroid_scatter": lambda r: 1.0 / max(g(r, "centroid_scatter"), 1e-6),
    "1/sinusoidity (shape)": lambda r: 1.0 / max(g(r, "sinusoidity"), 1e-6),
    "1/vingness (V vs U)": lambda r: -g(r, "vingness"),
    "1/secondary_ratio": lambda r: -g(r, "secondary_ratio"),
}

for name, f in tests.items():
    p = np.array([f(r) for r in pos], float)
    n = np.array([f(r) for r in neg], float)
    print(f"  {name:26s} AUC={auc(p, n):.3f}")


def combo(r):
    v = [g(r, "snr_power"), 1.0 / max(g(r, "odd_even_ratio"), 1e-6),
         1.0 / max(g(r, "centroid_scatter"), 1e-6),
         g(r, "depth_ppm") / g(r, "koi_depth_ppm")]
    if not all(np.isfinite(v)):
        return np.nan
    return float(np.prod(v))


p = np.array([combo(r) for r in pos], float)
n = np.array([combo(r) for r in neg], float)
print(f"\n  {'COMBINED (product)':26s} AUC={auc(p, n):.3f}")
