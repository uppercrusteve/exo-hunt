"""Batch sweep over a list of Kepler targets.

Resumable: target ids already present in the output CSV are skipped, so a run
can be interrupted and restarted. Writes one row per star.

Usage:
    python sweep.py --list data/neg_false_positive.csv --label fp --jobs 4
"""
import argparse
import csv
import os
import sys
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

FIELDS = ["kepid", "label", "koi_period", "koi_depth_ppm", "status", "period",
          "duration", "snr_power", "depth_ppm", "depth_odd_ppm", "depth_even_ppm",
          "odd_even_ratio", "sinusoidity", "vingness", "secondary_ratio", "oot_slope", "centroid_scatter", "centroid_pix_2d", "n_transits", "n_points", "alias_resolved", "note"]


def load_list(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["kepid"] for r in csv.DictReader(fh) if r.get("status") == "ok"}


def run_one(job):
    import detect

    kepid, label, koi_period, koi_depth = job
    row = {k: "" for k in FIELDS}
    row.update(kepid=kepid, label=label, koi_period=koi_period,
               koi_depth_ppm=koi_depth)
    try:
        lc = detect.load_lightcurve(f"kplr{kepid}")
        if lc is None:
            row["status"] = "nodata"
            return row
        t_all, f_all, _ = detect._arrays(lc)
        peaks = detect.bls_search(lc, top_k=6)
        if not peaks:
            row["status"] = "nopeak"
            return row
        b = peaks[0]
        d_odd, d_even = b["depth_odd"], b["depth_even"]
        row.update(
            status="ok",
            period=f"{b['period']:.6f}",
            duration=f"{b['duration']:.4f}",
            snr_power=f"{b['snr_power']:.1f}",
            depth_ppm=f"{b['depth'] * 1e6:.1f}",
            depth_odd_ppm=f"{d_odd * 1e6:.1f}",
            depth_even_ppm=f"{d_even * 1e6:.1f}",
            odd_even_ratio=f"{abs(d_odd - d_even) / max(abs(d_odd), abs(d_even), 1e-12):.4f}",
            n_transits=f"{b['n_transits']:.1f}",
            n_points=b["n_points"],
            alias_resolved=bool(b.get("alias_resolved", False)),
        )
        st = detect.shape_test(t_all, f_all, b["period"], b["transit_time"],
                               b.get("duration_eff", b["duration"]))
        if st:
            row["sinusoidity"] = f"{st['sinusoidity']:.3f}"
            row["oot_slope"] = f"{st['oot_slope']:.3f}"
        es = detect.eclipse_shape(t_all, f_all, b["period"], b["transit_time"], b["duration_eff"])
        if es:
            row["vingness"] = f"{es['vingness']:.3f}"
            row["secondary_ratio"] = f"{es['secondary_ratio']:.4f}"
        c = detect.centroid_test(lc, b["period"], b["transit_time"], b["duration"])
        if c:
            row["centroid_scatter"] = f"{c['centroid_scatter']:.2f}"
            row["centroid_pix_2d"] = f"{c['centroid_pix_2d']:.5f}"
    except Exception as e:
        row["status"] = "error"
        row["note"] = f"{type(e).__name__}: {str(e)[:80]}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="out/sweep.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = load_list(args.list)
    skip = done_ids(args.out)
    jobs = [(r["kepid"], args.label, r.get("koi_period", ""), r.get("koi_depth", ""))
            for r in rows if r["kepid"] not in skip]
    print(f"{len(rows)} targets, {len(skip)} already done, {len(jobs)} to run")

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        with Pool(args.jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
                w.writerow(row)
                fh.flush()
                print(f"[{i}/{len(jobs)}] {row['kepid']} {row['status']} "
                      f"snr={row['snr_power'] or '-'} oe={row['odd_even_ratio'] or '-'}",
                      flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
