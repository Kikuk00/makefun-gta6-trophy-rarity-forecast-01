"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate on
Exophase at the case's target, as five quantiles (5/25/50/75/95%), scored by
mean pinball loss on the logit scale.

Model (standard library only, deterministic, no network, no wall clock):

A trophy's rate is earners / players-tracked, and both move after launch:
Exophase's tracked-player count grows as new players sync (which dilutes
every rate down), while players play through the game and earn trophies
(which pushes rates back up).  Persistence -- repeating the last read --
captures neither move, so this forecaster fits each trophy's trajectory
and extrapolates it to the target, in three ways, and takes the middle
answer in logit space:

1. a saturating curve fitted to the trophy's own rate series: the rate
   follows A + B*exp(-t/tau) in hours since the release, with tau chosen
   by grid search (a trophy's share of players rises towards an asymptote
   as players progress, or falls towards one as new players dilute it);
2. the same saturating form fitted separately to the trophy's earners
   E(t) = rate * tracked / 100 and to the tracked count N(t), then the
   forecast rate is E(T) / N(T) at the target -- this is the form that
   can keep falling when a surge of new players outruns progression;
3. a log-time linear trend: a least-squares line of logit(rate) against
   log(1 + hours) over the last 24 hours of reads, read at the target --
   a form-agnostic trend that keeps moving where the saturating curves
   level off.

The median of the three is the forecast median: when the fitted forms
agree, the trend is trusted; when they disagree, the answer lands
between them instead of chasing either extrapolation.  Persistence is
only the fallback when no trend could be fitted, not a vote.  A trend
is only used when it beats a flat line clearly on the observed reads,
so noisy trophies fall back towards persistence instead of fitting
noise.

Reads are collapsed first: consecutive identical reads are one Exophase
refresh, not new information, so each distinct level enters the fit once,
at the first read that showed it.

Spreads are honest about what moves rates: each trophy's own residual
volatility around its fitted curve, stretched by how far past the
observed window the target sits, over a floor that covers the refresh
timing and the shape of the rise being wrong.  There is deliberately no
random-walk term: the steps between Exophase refreshes are mostly the
trend itself, not noise, so a random-walk spread would double-count the
move the median already forecasts.  A trophy whose history jumped
around gets a wide band; a steady one gets a narrow band.

Special cases:
* a flat history (a single distinct read, as on launch night): no trend
  can be fitted, so the median allows for one Exophase refresh landing
  inside the horizon -- tracked players jump and rates dilute down -- and
  the spread is wide enough to cover a refresh of the size launch night
  saw;
* a trophy at 100% whose earners equal the tracked count in every read
  stays at 100% (every tracked player holds it the moment they sync);
* a trophy at the 0.01% floor, null rates in early reads, and a history
  with a single read are all handled by falling back towards persistence
  with a modest spread instead of crashing.

I/O contract: read every <id>.in in $MAKEFUN_INPUT_DIR (else /input),
write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
per trophy, quantiles in percent rounded to 4 decimals (half up).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Grid of saturating-curve timescales, in hours (2 h .. ~1200 h).
TAU_GRID = [2.0 * (600.0 ** (k / 47.0)) for k in range(48)]
# A trend must beat the flat line by this factor of squared error to be used.
TREND_GAIN = 0.90
# Widest logit move a median may make away from the last read.
MAX_MOVE = 4.0
# Random-walk spread, logit units per square-root day (the house baseline's).
RW_SIGMA = 0.5
# On a flat history, one refresh inside the horizon is assumed to dilute
# rates by about this factor (launch night's refresh took tracked players
# 638 -> 3,868 while earners barely moved).
FLAT_DILUTION = 2.5
FLAT_SIGMA = 1.3


def _parse_time(text):
    """ISO-8601 timestamp (with a Z suffix) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def _exp_fit(ts, ys, ws, lo, hi, rising_only=False):
    """Fit y = A + B*exp(-t/tau) by grid search over tau, weighted least
    squares for (A, B) at each tau.

    Returns (A, B, tau, sse, sse_flat): the best fit and the flat line's
    squared error, for deciding whether the trend is worth anything.
    A is clamped into [lo, hi]; with rising_only, B is clamped to <= 0
    (the curve may only rise towards its asymptote A).
    """
    sw = sum(ws)
    mean = sum(w * y for w, y in zip(ws, ys)) / sw
    sse_flat = sum(w * (y - mean) ** 2 for w, y in zip(ws, ys))
    best = None
    for tau in TAU_GRID:
        ex = [math.exp(-t / tau) for t in ts]
        sew = sum(w * e for w, e in zip(ws, ex))
        se2 = sum(w * e * e for w, e in zip(ws, ex))
        sey = sum(w * e * y for w, e, y in zip(ws, ex, ys))
        sy = sum(w * y for w, y in zip(ws, ys))
        det = sw * se2 - sew * sew
        if abs(det) < 1e-12 or se2 <= 0.0:
            continue
        a = (sy * se2 - sew * sey) / det
        a = min(hi, max(lo, a))
        # Refit B given the clamped A.
        b = sum(w * e * (y - a) for w, e, y in zip(ws, ex, ys)) / se2
        if rising_only and b > 0.0:
            b = 0.0
            a = mean
        sse = sum(w * (y - (a + b * e)) ** 2 for w, e, y in zip(ws, ex, ys))
        if best is None or sse < best[3]:
            best = (a, b, tau, sse)
    if best is None:
        return (mean, 0.0, None, sse_flat, sse_flat)
    return best + (sse_flat,)


def _distinct_reads(history):
    """Collapse consecutive identical reads: one Exophase refresh, one row.

    Keeps the first read that shows each new level, oldest first.
    """
    seen = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            seen.append(read)
            prev_key = key
    return seen


def _quantiles(median_logit, sigma):
    qs = []
    for z in Z:
        q = _round4(_clamp_pct(_inv_logit_pct(median_logit + z * sigma)))
        qs.append(q)
    # Rounding can only tie neighbours, never invert them, but enforce it.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def _forecast_trophy(idx, case, times, tracked, rates_by_read, horizon_days, span_days, target_t):
    """Forecast one trophy: returns its five quantiles."""
    # The trophy's own series over the distinct reads, skipping nulls.
    series = []
    for j, t in enumerate(times):
        r = rates_by_read[j][idx]
        if r is not None:
            series.append((t, tracked[j], r))
    if not series:
        # Never seen: a low rate, honestly uncertain.
        return _quantiles(_logit(0.5), 2.0)
    last_rate = series[-1][2]
    # Trophy at 100% in every read: every tracked player holds it on
    # sync, so it stays at 100%.  Checked before the short-history
    # fallback, so a flat history cannot dilute a trophy nobody can miss.
    if last_rate >= 99.99 and all(r >= 99.99 for _, _, r in series):
        return [99.95, 99.99, 100, 100, 100]
    if len(series) < 5 or len(times) < 3:
        # Too little to fit a trend: persistence, with a spread that
        # admits a refresh inside the horizon.
        if len(times) < 2:
            return _quantiles(_logit(last_rate / FLAT_DILUTION), FLAT_SIGMA)
        return _quantiles(_logit(last_rate), max(RW_SIGMA * math.sqrt(horizon_days), 0.6))

    ts = [t for t, _, _ in series]
    ns = [n for _, n, _ in series]
    shares = [r / 100.0 for _, _, r in series]

    # Candidate 1: saturating curve on the rate itself (fitted in share
    # units 0..1, converted back to percent for the forecast).
    a_r, b_r, tau_r, sse_r, flat_r = _exp_fit(ts, shares, ns, 0.0001, 0.9999)
    if tau_r is not None and sse_r < TREND_GAIN * flat_r:
        cand1 = 100.0 * (a_r + b_r * math.exp(-target_t / tau_r))
    else:
        cand1 = None
    # Residual volatility of the rate fit, in logit space, for the spread.
    if cand1 is not None:
        fit_vals = [a_r + b_r * math.exp(-t / tau_r) for t in ts]
    else:
        wmean = sum(n * s for n, s in zip(ns, shares)) / sum(ns)
        fit_vals = [wmean] * len(ts)
    resid2 = sum(
        n * (_logit(100.0 * s) - _logit(100.0 * max(0.0001, min(0.9999, f)))) ** 2
        for n, s, f in zip(ns, shares, fit_vals)
    )
    resid = math.sqrt(resid2 / sum(ns))

    # Candidate 2: saturating curves on earners and on tracked players.
    earners = [s * n for s, n in zip(shares, ns)]
    a_e, b_e, tau_e, sse_e, flat_e = _exp_fit(ts, earners, [1.0] * len(ts), 0.0, float("inf"), rising_only=True)
    a_n, b_n, tau_n, sse_n, flat_n = _exp_fit(ts, [float(n) for n in ns], [1.0] * len(ts), float(max(ns)), float("inf"), rising_only=True)
    cand2 = None
    if tau_e is not None and tau_n is not None:
        e_hat = a_e + b_e * math.exp(-target_t / tau_e)
        n_hat = a_n + b_n * math.exp(-target_t / tau_n)
        if n_hat > 0 and e_hat >= 0:
            cand2 = e_hat / n_hat * 100.0

    # Candidate 3: persistence.
    cand3 = last_rate

    # Candidate 4: the log-time linear trend -- a least-squares line of
    # logit(rate) against log(1 + hours) over the last 24 hours of reads,
    # read at the target.  A form-agnostic trend: it keeps rising or
    # falling in log-time where the saturating curves level off, so it
    # disagrees with them exactly when the shape of the rise is unclear,
    # and the median below then lands between the two views.
    cand4 = None
    last_t = ts[-1]
    pts4 = [(math.log1p(t), _logit(r)) for t, _, r in series if last_t - t <= 24.0]
    if len(pts4) >= 3:
        n4 = len(pts4)
        mx4 = sum(x for x, _ in pts4) / n4
        my4 = sum(y for _, y in pts4) / n4
        sxx4 = sum((x - mx4) ** 2 for x, _ in pts4)
        if sxx4 > 0.0:
            slope4 = sum((x - mx4) * (y - my4) for x, y in pts4) / sxx4
            inter4 = my4 - slope4 * mx4
            cand4 = _inv_logit_pct(inter4 + slope4 * math.log1p(target_t))

    logits = []
    for cand in (cand1, cand2, cand4):
        if cand is not None:
            logits.append(_logit(cand))
    if not logits:
        logits.append(_logit(cand3))
    logits.sort()
    # Median of the trend estimators: the middle value for three, the
    # mean of the middle two for two.  Persistence is the fallback when
    # no trend could be fitted, not a vote: the trends already straddle
    # it, and a sceptical vote only drags the median towards a forecast
    # every trend disagrees with.
    median_logit = logits[len(logits) // 2] if len(logits) % 2 else (
        logits[len(logits) // 2 - 1] + logits[len(logits) // 2]
    ) / 2.0

    # Guard the move away from the last read.
    last_logit = _logit(last_rate)
    move = median_logit - last_logit
    move = min(MAX_MOVE, max(-MAX_MOVE, move))
    median_logit = last_logit + move

    # Spread: this trophy's own residual volatility around its fitted
    # curve, stretched by how far past the observed window the target
    # sits, over a floor that covers the refresh timing and the shape of
    # the rise being wrong.  There is deliberately no random-walk term
    # here: the steps between Exophase refreshes are mostly the trend
    # itself, not noise, so a random-walk spread would double-count the
    # move the median already forecasts and pay for width at every
    # trophy.  The flat-history fallback below keeps a random-walk-style
    # spread, where no trend exists to explain the steps.
    stretch = math.sqrt(max(horizon_days, 0.1) / max(span_days, 0.1))
    vol = min(0.8, resid * stretch)
    sigma = math.sqrt(vol * vol + 0.35 * 0.35)
    sigma = min(2.2, max(sigma, 0.2))
    return _quantiles(median_logit, sigma)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    # Hours since the release day began; the axis the trends are fitted on.
    try:
        release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    except (KeyError, TypeError, ValueError):
        release = _parse_time(history[0]["at"]) if history else datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = _parse_time(case["target"])
    target_t = max((target - release).total_seconds() / 3600.0, 0.0)

    if not history:
        # No reads at all: say so honestly, one low wide row per trophy.
        return {"trophies": [{"id": t["id"], "q": _quantiles(_logit(1.0), 2.0)} for t in case["trophies"]]}

    reads = _distinct_reads(history)
    times = [max((_parse_time(r["at"]) - release).total_seconds() / 3600.0, 0.0) for r in reads]
    tracked = [max(int(r["players_tracked"]), 1) for r in reads]
    rates_by_read = [list(r["rates"]) for r in reads]
    span_days = max((times[-1] - times[0]) / 24.0, 1.0 / 24.0) if len(times) > 1 else 1.0 / 24.0

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rows.append(
            {
                "id": trophy["id"],
                "q": _forecast_trophy(
                    idx, case, times, tracked, rates_by_read, horizon_days, span_days, target_t
                ),
            }
        )
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
