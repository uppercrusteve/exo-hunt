"""Phase-fold a target on the detected ephemeris and render a PNG.

Binned over the raw scatter so a weak signal is visible without overplotting
tens of thousands of points.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kepid")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    lc = detect.load_lightcurve(f"kplr{args.kepid}")
    if lc is None:
        print("no data"); return 1
    b = detect.bls_search(lc, top_k=3)[0]
    t, f, _ = detect._arrays(lc)

    ph, flux = detect.phase_fold(t, f, b["period"], b["transit_time"])
    order = np.argsort(ph)
    ph, flux = ph[order], flux[order]

    nb = 120
    edges = np.linspace(-0.5, 0.5, nb + 1)
    idx = np.digitize(ph, edges)
    bx, by, be = [], [], []
    for i in range(1, nb + 1):
        m = idx == i
        if m.sum() >= 2:
            bx.append(0.5 * (edges[i - 1] + edges[i]))
            by.append(np.median(flux[m]))
            be.append(np.std(flux[m]) / np.sqrt(m.sum()))

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(ph, flux, ".", ms=1.2, color="0.72", alpha=0.5, rasterized=True)
    ax.errorbar(bx, by, yerr=be, fmt="o", ms=3.5, color="#c0392b", capsize=0, lw=0.9)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(f"phase  (P = {b['period']:.5f} d)")
    ax.set_ylabel("normalized flux")
    ax.set_title(
        f"KIC {args.kepid}   depth {b['depth']*1e6:.0f} ppm   "
        f"SNR {b['snr_power']:.0f}   odd/even {abs(b['depth_odd']-b['depth_even'])/max(b['depth_odd'],b['depth_even'])*100:.2f}%"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(args.out, f"fold_{args.kepid}.png")
    fig.savefig(path)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
