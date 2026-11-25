"""Core transit detection: load a Kepler light curve, BLS period search, vet.

Two things this module deliberately does itself instead of trusting astropy:
  * depth measurement - astropy's compute_stats returns depth *summed over
    transits*, not a fractional depth (verified: depth/n_transits recovers the
    injected value). We fold and measure in/out medians instead.
  * period disambiguation - BLS happily reports P/2. We test odd/even transit
    depths and promote to 2P when the half-period hypothesis splits them.
"""
import warnings

import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")


def load_lightcurve(target, mission="Kepler", cadence="long"):
    """Download + clean a light curve. Returns None when no usable data."""
    import lightkurve as lk

    sr = lk.search_lightcurve(target, mission=mission, cadence=cadence)
    if len(sr) == 0:
        return None
    lcc = sr.download_all()
    if lcc is None:
        return None
    lc = lcc.stitch() if hasattr(lcc, "stitch") else lcc
    lc = lc.remove_nans().normalize()

    # Kepler quality flags carry the real systematics (momentum dumps, SAA
    # crossings, cosmic rays). Drop any flagged cadence before touching flux.
    if "quality" in lc.colnames:
        q = np.asarray(lc.quality)
        lc = lc[q == 0]

    # No sigma clip: the flux std of a star hosting a 1.5% transit is ~0.0018,
    # so a 3-4 sigma clip truncates the transit itself. PDCSAP is already
    # systematics-corrected and the quality mask handles the rest.
    if len(lc) < 500:
        return None
    return lc


def _arrays(lc):
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    dy = np.asarray(lc.flux_err.value, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(dy) & (dy > 0)
    t, f, dy = t[ok], f[ok], dy[ok]
    order = np.argsort(t)
    return t[order], f[order], dy[order]


def measure(t, f, period, t0, duration, offset=0.0):
    """Fold on the ephemeris and measure depth + odd/even consistency.

    depth is (out-of-transit median - in-transit median) / out-of-transit
    median, i.e. a genuine fractional depth. odd/even are the same quantity
    restricted to alternating transits: unequal values flag an eclipsing
    binary rather than a planet.
    """
    t0 = t0 + offset * period
    n = np.round((t - t0) / period)
    phase = (t - t0) / period - n

    # BLS's likelihood maximiser prefers the shortest duration in the grid,
    # which on 29.4 min Kepler cadence leaves ~2 points in transit and makes
    # every median shaky. Re-optimise outward and keep the deepest box.
    best = None
    for scale in (1.0, 1.3, 1.7, 2.2, 2.8):
        dur = duration * scale
        half = max(dur / period / 2.0, 1.5 / 1440.0 / period)
        in_tr = np.abs(phase) < half
        out_tr = (np.abs(phase) > 1.5 * half) & (np.abs(phase) < 0.45)
        if out_tr.sum() < 20 or in_tr.sum() < 5:
            continue
        base = float(np.median(f[out_tr]))
        if base <= 0:
            continue
        depth = float(1.0 - np.median(f[in_tr]) / base)
        if best is None or depth > best[0]:
            best = (depth, dur, in_tr, out_tr, base, half)

    if best is None:
        return None

    depth, dur, in_tr, out_tr, base, half = best

    def _depth(mask):
        if mask.sum() < 3:
            return float("nan")
        return float(1.0 - np.median(f[mask]) / base)

    return {
        "depth": depth,
        "depth_odd": _depth(in_tr & (np.abs(n % 2) == 1)),
        "depth_even": _depth(in_tr & (np.abs(n % 2) == 0)),
        "duration_eff": dur,
        "n_in_transit": int(in_tr.sum()),
        "n_out_transit": int(out_tr.sum()),
    }


def bls_search(lc, min_period=1.0, max_period=15.0, n_periods=20000,
               durations=(0.04, 0.07, 0.10, 0.15, 0.22, 0.30), top_k=12,
               n_refine=4000):
    """BLS period search; return the top_k de-duplicated, locally refined peaks.

    Two details matter more than they look:
      * the period grid is logarithmic. On a linear grid the relative step is
        coarser at long periods, so a 2P harmonic survives phase drift better
        than the true period and wins. Verified on TrES-2: linear grid returned
        2P, log grid returns P.
      * every surviving peak is refined on a narrow local grid, so the value
        reported is a real local maximum and not an artifact of grid spacing.

    Significance is (peak - median) / MAD of the power spectrum, so values are
    comparable across stars of different noise levels.
    """
    t, f, dy = _arrays(lc)
    if len(t) < 500:
        return []

    bls = BoxLeastSquares(t * u.day, f, dy)
    dur_q = np.asarray(durations) * u.day
    periods = np.exp(np.linspace(np.log(min_period), np.log(max_period), n_periods))
    res = bls.power(periods, dur_q)
    power = np.asarray(res.power, dtype=float)

    med = float(np.nanmedian(power))
    mad = float(np.nanmedian(np.abs(power - med))) or 1e-12

    peaks, taken = [], []
    for i in np.argsort(power)[::-1]:
        p = float(res.period[i].to_value("day"))
        # skip peaks within 1% of an already-accepted period
        if any(abs(p - q) / q < 0.01 for q in taken):
            continue
        taken.append(p)
        pk = {"period": p,
              "duration": float(res.duration[i].to_value("day")),
              "transit_time": float(res.transit_time[i].to_value("day"))}
        _refine(bls, pk, float(periods[i]), dur_q)
        peaks.append(pk)
        if len(peaks) >= top_k:
            break

    baseline = t[-1] - t[0]
    for pk in peaks:
        pk["power"] = float(bls.power(np.array([pk["period"]]), dur_q).power[0])
        pk["snr_power"] = (pk["power"] - med) / mad
        pk["baseline_days"] = float(baseline)
        pk["n_transits"] = float(baseline / pk["period"])
        pk["n_points"] = int(len(t))
        m = measure(t, f, pk["period"], pk["transit_time"], pk["duration"])
        if m:
            pk.update(m)

    peaks.sort(key=lambda z: z["power"], reverse=True)

    # Harmonic ambiguity. BLS lands on whichever of P, P/2 or 2P maximises
    # likelihood, which is a grid artefact as much as a physical statement.
    # Odd/even depth cannot separate P from 2P - the 2P solution simply uses
    # half the events and looks cleaner - so test the physics instead: if a
    # full-depth dip also sits at phase 0.5, the true period is half.
    for pk in peaks:
        pk = _fix_harmonic(t, f, pk, min_period)
    return peaks


def _fix_harmonic(t, f, pk, min_period, max_iters=3):
    """Double if alternate transits disagree, halve if a secondary is present."""
    base = pk["duration"]
    for _ in range(max_iters):
        d2 = 2.0 * pk["period"]
        m2 = measure(t, f, d2, pk["transit_time"], base)
        if not (m2 and _alias_better(pk, m2)):
            break
        pk["period"] = d2
        pk["n_transits"] = pk["baseline_days"] / d2
        pk.update(m2)
        pk["alias_resolved"] = True

    for _ in range(max_iters):
        half = 0.5 * pk["period"]
        if half < min_period:
            break
        mh = measure(t, f, pk["period"], pk["transit_time"], base, offset=0.5)
        if not (mh and mh["depth"] > 0.6 * pk["depth"]):
            break
        m1 = measure(t, f, half, pk["transit_time"], base)
        if not m1 or m1["depth"] <= 0:
            break
        pk["period"] = half
        pk["n_transits"] = pk["baseline_days"] / half
        pk.update(m1)
        pk["alias_resolved"] = True
    return pk




def _split(m):
    d1, d2 = m.get("depth_odd"), m.get("depth_even")
    if not np.isfinite([d1, d2]).all() or d1 <= 0 or d2 <= 0:
        return float("inf")
    return abs(d1 - d2) / max(abs(d1), abs(d2))


def _pick_harmonic(t, f, pk, min_period, max_period):
    """Choose among P/2, P, 2P for one peak. Returns (multiplier, period, m)."""
    cands = []
    for mult in (0.5, 1.0, 2.0):
        p = pk["period"] * mult
        if not (min_period <= p <= max_period):
            continue
        m = measure(t, f, p, pk["transit_time"], pk["duration"])
        if m and m["depth"] > 0:
            cands.append((mult, p, m))
    if not cands:
        return None

    base_depth = max(c[2]["depth"] for c in cands)
    # a harmonic that dilutes the depth by >25% is the wrong one
    keep = [c for c in cands if c[2]["depth"] >= 0.75 * base_depth]
    if not keep:
        keep = cands
    # Rank on odd/even agreement (bucketed, so two clean solutions tie and the
    # decision falls through to period), then prefer the shorter period: the
    # shorter baseline per transit means more events and a tighter constraint.
    def rank(c):
        sp = _split(c[2])
        tier = 0 if sp < 0.05 else (1 if sp < 0.15 else 2)
        return (tier, c[1])

    keep.sort(key=rank)
    return keep[0]


def _refine(bls, pk, p_guess, dur_q):
    """Sharpen a peak on a narrow grid; writes period/duration/transit_time."""
    local = np.linspace(p_guess * 0.997, p_guess * 1.003, 4000)
    r = bls.power(local, dur_q)
    pw = np.asarray(r.power, dtype=float)
    j = int(np.nanargmax(pw))
    pk["period"] = float(r.period[j].to_value("day"))
    pk["duration"] = float(r.duration[j].to_value("day"))
    pk["transit_time"] = float(r.transit_time[j].to_value("day"))


def _alias_better(one, two):
    """Is the 2P hypothesis better than the P/2 one?

    Accept when the doubled period removes the odd/even mismatch, i.e. the
    current fit splits alternate transits (sign of a half-period alias).
    """
    d1, e1 = one.get("depth_odd"), one.get("depth_even")
    d2, e2 = two.get("depth_odd"), two.get("depth_even")
    if not all(np.isfinite([d1, e1, d2, e2])) or d1 <= 0 or d2 <= 0:
        return False
    split1 = abs(d1 - e1) / max(abs(d1), abs(e1))
    split2 = abs(d2 - e2) / max(abs(d2), abs(e2))
    # promoted only if 2P is genuinely more consistent, and the depth survives.
    # The ratio window is wide on purpose: at P/2 the box catches alternate
    # transits only, so the measured depth is roughly halved by construction.
    return split2 < 0.5 * split1 and 0.3 < d2 / d1 < 3.5


def phase_fold(t, f, period, transit_time):
    ph = ((t - transit_time) / period + 0.5) % 1.0 - 0.5
    return ph, f


def _detrend(y, t, window_days):
    """Subtract a running-median trend sampled on a fixed day grid."""
    lo, hi = float(np.min(t)), float(np.max(t))
    edges = np.arange(lo, hi + window_days, window_days)
    if edges.size < 3:
        return y
    idx = np.digitize(t, edges)
    centers, meds = [], []
    for i in range(1, edges.size):
        m = idx == i
        if m.sum() >= 3:
            centers.append(0.5 * (edges[i - 1] + edges[i]))
            meds.append(float(np.median(y[m])))
    if len(centers) < 3:
        return y
    return y - np.interp(t, centers, meds)


def centroid_test(lc, period, t0, duration):
    """Does the photocentre move during transit, and in which direction?

    A background eclipsing binary diluted by the target star pulls the
    flux-weighted centroid toward the eclipse source. A real planet keeps it
    put. Returns the shift in units of its own uncertainty, plus the raw arc.

    This is the main prior-free veto: unlike depth-vs-catalog comparisons it
    needs no advance knowledge of the star.
    """
    if "mom_centr1" not in lc.colnames:
        return None
    try:
        cx = np.asarray(lc.mom_centr1.value, dtype=float)
        cy = np.asarray(lc.mom_centr2.value, dtype=float)
        ex = np.asarray(lc.mom_centr1_err.value, dtype=float)
        ey = np.asarray(lc.mom_centr2_err.value, dtype=float)
        t = np.asarray(lc.time.value, dtype=float)
    except Exception:
        return None

    ok = np.isfinite(t) & np.isfinite(cx) & np.isfinite(cy) & np.isfinite(ex) & np.isfinite(ey)
    if ok.sum() < 500:
        return None
    t, cx, cy, ex, ey = t[ok], cx[ok], cy[ok], ex[ok], ey[ok]

    n = np.round((t - t0) / period)
    phase = (t - t0) / period - n
    half = max(2.0 * duration / period / 2.0, 1.5 / 1440.0 / period)
    in_tr = np.abs(phase) < half
    out_tr = (np.abs(phase) > 1.5 * half) & (np.abs(phase) < 0.45)
    if in_tr.sum() < 5 or out_tr.sum() < 20:
        return None

    # Kepler's centroid wanders by ~1 pixel over a campaign from pointing
    # drift and velocity aberration, while a blend-induced shift is ~0.003
    # pixel. Detrend with a 1-day running median before measuring, otherwise
    # the low-frequency wander buries the transit-coherent signal entirely.
    cx = _detrend(cx, t, 1.0)
    cy = _detrend(cy, t, 1.0)

    dx = float(np.median(cx[in_tr]) - np.median(cx[out_tr]))
    dy = float(np.median(cy[in_tr]) - np.median(cy[out_tr]))
    sx = float(np.std(cx[out_tr]))
    sy = float(np.std(cy[out_tr]))
    # offset in units of the out-of-transit centroid scatter: dimensionless,
    # comparable between stars, and insensitive to the number of in-transit
    # points (which the formal error divides by and makes incomparable).
    sig = float(np.hypot(dx / sx, dy / sy)) if sx > 0 and sy > 0 else float("nan")
    return {"centroid_pix_2d": float(np.hypot(dx, dy)), "centroid_scatter": sig}


def shape_test(t, f, period, t0, duration):
    """How transit-like is the folded shape?

    A planet transit is a flat-bottomed box: outside the transit the light
    curve is flat, and the drop occupies a small phase window. A pulsating or
    contact-binary star is a sine wave, so the "out of transit" baseline is
    tilted by an amplitude comparable to the depth itself.

    Returns:
      sinusoidity  - out-of-transit peak-to-peak / transit depth.
                     ~0 for a box, ~1 for a sine.
      oot_slope    - fitted |slope| across the out-of-transit phase,
                     normalised by depth (in units of depth per unit phase).
    """
    n = np.round((t - t0) / period)
    phase = (t - t0) / period - n
    half = max(duration / period / 2.0, 1.5 / 1440.0 / period)
    in_tr = np.abs(phase) < half
    oot = (np.abs(phase) > 2.5 * half) & (np.abs(phase) < 0.45)
    if in_tr.sum() < 5 or oot.sum() < 50:
        return None

    base = float(np.median(f[oot]))
    depth = float(1.0 - np.median(f[in_tr]) / base)
    if depth <= 0:
        return None

    amp = float(np.percentile(f[oot], 95) - np.percentile(f[oot], 5)) / base
    p = phase[oot]
    cfit = np.polyfit(p, f[oot] / base, 1)
    return {
        "sinusoidity": amp / depth,
        "oot_slope": float(abs(cfit[0]) * 0.9 / depth),
    }


def eclipse_shape(t, f, period, t0, duration):
    """Distinguish a V-shaped stellar eclipse from a U-shaped planetary one,
    and measure any secondary eclipse half a period away.

    Stellar eclipses are limb-darkened V shapes with a comparable secondary;
    planet transits are flat-bottomed (the planet is small and opaque) and the
    secondary is only reflected/emitted light, typically <100 ppm.
    """
    t0s = t0 + 0.5 * period
    n = np.round((t - t0) / period)
    phase = np.abs((t - t0) / period - n)
    half = max(duration / period / 2.0, 1.5 / 1440.0 / period)

    in_tr = phase < half
    oot = (phase > 2.5 * half) & (phase < 0.45)
    if in_tr.sum() < 9 or oot.sum() < 50:
        return None
    base = float(np.median(f[oot]))
    if base <= 0:
        return None

    core = phase < half / 3.0
    wings = (phase >= half / 3.0) & in_tr
    if core.sum() < 3 or wings.sum() < 3:
        return None
    d_core = 1.0 - float(np.median(f[core])) / base
    d_wing = 1.0 - float(np.median(f[wings])) / base
    if d_core <= 0:
        return None
    vingness = float(np.clip(1.0 - d_wing / d_core, -1, 2))

    # secondary: same window, half a period later
    ns = np.round((t - t0s) / period)
    phs = np.abs((t - t0s) / period - ns)
    sec = phs < half
    secondary = float("nan")
    if sec.sum() >= 5:
        secondary = max(0.0, 1.0 - float(np.median(f[sec])) / base) / d_core

    return {"vingness": vingness, "secondary_ratio": secondary}
