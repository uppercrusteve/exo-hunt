"""Self-check against TrES-2b (KIC 11446443).

Ground truth: P = 2.47061 d, depth ~1.5 %. The pipeline must recover the
period to well under the 29.4 min cadence, keep the depth near truth, and
report consistent odd and even depths. Run this after touching detect.py.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import detect

KIC = "11446443"
TRUE_P = 2.47061
TRUE_DEPTH = 0.015
PERIOD_TOL_MIN = 5.0


def main():
    lc = detect.load_lightcurve(f"kplr{KIC}")
    if lc is None:
        print("FAIL: no light curve"); return 1
    t, f, _ = detect._arrays(lc)
    b = detect.bls_search(lc, top_k=3)[0]

    err_min = abs(b["period"] - TRUE_P) * 1440.0
    depth_err = abs(b["depth"] - TRUE_DEPTH) / TRUE_DEPTH
    split = detect._split(b)
    sin = detect.shape_test(t, f, b["period"], b["transit_time"], b["duration_eff"])

    print(f"period   {b['period']:.5f} d   (truth {TRUE_P}, err {err_min:.2f} min)")
    print(f"depth    {b['depth']*100:.3f} %  (truth {TRUE_DEPTH*100:.2f} %, rel err {depth_err*100:.1f} %)")
    print(f"odd/even split {split*100:.2f} %")
    print(f"sinusoidity    {sin['sinusoidity']:.3f}")

    ok = True
    for name, cond in (("period", err_min < PERIOD_TOL_MIN),
                       ("depth", depth_err < 0.35),
                       ("odd/even", split < 0.10),
                       ("shape is box-like", sin["sinusoidity"] < 0.5)):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
