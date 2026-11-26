"""Filter a completed sweep into a candidate list under an explicit threshold.

Kept separate from sweep.py so the same scan can be re-cut at different
thresholds without re-downloading anything. Reports what the cut costs, not
just what it keeps.
"""
import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

FIELDS = ["kepid", "label", "status", "period", "snr_power", "depth_ppm",
          "odd_even_ratio", "sinusoidity", "vingness", "secondary_ratio",
          "centroid_scatter", "n_transits", "note"]

# Measured on the truth-labelled control set; see README. `strict` is the
# highest-purity operating point found, `default` follows the SNR>1000 brief.
PRESETS = {
    "default": dict(min_snr=1000.0, max_sinusoidity=1.5, max_vingness=9.9,
                    max_secondary=9.9),
    "strict": dict(min_snr=8000.0, max_sinusoidity=1.5, max_vingness=0.3,
                   max_secondary=0.02),
}


def num(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="out/sweep.csv")
    ap.add_argument("--out", default="out/candidates.csv")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--min-snr", type=float)
    ap.add_argument("--max-sinusoidity", type=float)
    ap.add_argument("--max-vingness", type=float)
    ap.add_argument("--max-secondary", type=float)
    args = ap.parse_args()

    th = dict(PRESETS[args.preset]) if args.preset else dict(PRESETS["default"])
    for k in ("min_snr", "max_sinusoidity", "max_vingness", "max_secondary"):
        v = getattr(args, k)
        if v is not None:
            th[k] = v

    rows = [r for r in csv.DictReader(open(args.sweep, encoding="utf-8"))
            if r["status"] == "ok"]

    def keep(r):
        return (num(r, "snr_power") > th["min_snr"]
                and num(r, "sinusoidity") < th["max_sinusoidity"]
                and num(r, "vingness") < th["max_vingness"]
                and num(r, "secondary_ratio") < th["max_secondary"])

    sel = [r for r in rows if keep(r)]

    print(f"cut: SNR>{th['min_snr']:g}  sinusoidity<{th['max_sinusoidity']:g}  "
          f"vingness<{th['max_vingness']:g}  "
          f"secondary<{th['max_secondary']:g}")
    print(f"kept {len(sel)} of {len(rows)} scanned rows\n")
    tp = sum(1 for r in sel if r["label"] == "confirmed")
    fp = sum(1 for r in sel if r["label"] == "fp")
    quiet = sum(1 for r in sel if r["label"] == "quiet")
    n_conf = sum(1 for r in rows if r["label"] == "confirmed")
    n_fp = sum(1 for r in rows if r["label"] == "fp")
    n_q = sum(1 for r in rows if r["label"] == "quiet")
    if n_conf:
        print(f"  recall            {tp}/{n_conf} ({tp / n_conf * 100:.0f} %)")
    if n_fp:
        print(f"  false positives    {fp}/{n_fp} still pass")
    if n_q:
        print(f"  quiet stars        {quiet}/{n_q} wrongly kept")
    if tp + fp:
        print(f"  purity of output   {tp / (tp + fp) * 100:.0f} %  ({tp} real / {tp + fp} kept)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(sel, key=lambda z: -num(z, "snr_power")):
            w.writerow(r)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
