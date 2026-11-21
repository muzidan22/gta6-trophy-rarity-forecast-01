"""Forecasting engine for make.fun's gta6-trophy-rarity-forecast-01.

Standard library only: the grader runs ``main.py`` in a sandbox with no network and
nothing installed (``python3 -P -s /bot/main.py``), so this module must not import
pandas/numpy/scipy. It is imported by ``main.py`` (which restores its own directory
on ``sys.path`` because the grader's ``-P`` removes it).

The model
---------
Every trophy's Exophase earn rate ``r`` (percent of tracked players who earned it)
rises over the first week. We work on the logit scale, ``logit(r) = ln(r/(100-r))``,
which is unbounded and makes a miss from 1% to 2% cost about what a miss from 33% to
50% costs. For each trophy we fit, on the reads in the case's history (oldest first),

    logit(r) ~ a + b * x,      x = ln(1 + hours since release)

with exponentially decaying weights so the most recent reads drive the slope, then
read the line at the target's ``x`` and damp the extrapolated movement toward the
last observed read (a growth curve decelerates after launch). The predictive spread
combines the house random-walk band (``SIGMA_RW`` logit units per sqrt-day) with the
least-squares prediction error of the fit, and is turned into the five contest
quantiles ``q_tau = inv_logit(mu + z_tau * s)``.

Quantile levels, the logit definition and the random-walk spread match the kit's
rules and the house baseline, so a forecast here is directly comparable with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# --- Contest constants (must match RULES.txt) -------------------------------
LEVELS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
Z_LEVELS: tuple[float, ...] = (-1.645, -0.674, 0.0, 0.674, 1.645)
SIGMA_RW: float = 0.50  # house random-walk spread, logit units per sqrt-day

# --- Model constants --------------------------------------------------------
RELEASE_UTC: str = "2026-11-19T00:00:00Z"  # GTA VI PS5 release (game.release_date)
CUTOFF_UTC: str = "2026-11-21T11:30:00Z"   # the strict temporal cutoff (deadline)
TARGET_UTC: str = "2026-11-26T00:00:00Z"   # the final case's target
DAMPING: float = 0.85          # shrink the extrapolated logit move (deceleration)
HALF_LIFE_HOURS: float = 18.0  # weight decay for older reads
MIN_SPREAD: float = 0.30
MAX_SPREAD: float = 1.80


# --- Small numeric helpers --------------------------------------------------
def clamp_rate(x: float) -> float:
    """Clamp a percent rate to the logit-safe open interval (0.01, 99.99)."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    """logit of a percent rate, after clamping (0% and 100% have no logit)."""
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(z: float) -> float:
    """Inverse logit: a logit value back to a percent rate in (0, 100)."""
    return 100.0 / (1.0 + math.exp(-z))


def round4(x: float) -> float | int:
    """Round half up to 4 decimals, writing whole numbers as int (JS parity)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_utc(ts: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def hours_since(ts: str, origin: str = RELEASE_UTC) -> float:
    """Hours from ``origin`` to ``ts`` (both ISO 8601 UTC)."""
    return (parse_utc(ts) - parse_utc(origin)).total_seconds() / 3600.0


# --- Weighted least squares -------------------------------------------------
@dataclass(frozen=True)
class Line:
    """A weighted least-squares line ``y = a + b x`` with its fit quality."""

    a: float
    b: float
    x_bar: float
    sxx: float
    s_w: float
    resid_var: float

    def at(self, x: float) -> float:
        return self.a + self.b * x

    def se_pred(self, x: float) -> float:
        """Standard error of a prediction at ``x`` (0 when under-determined)."""
        if self.s_w <= 0.0 or self.sxx <= 0.0:
            return 0.0
        lever = 1.0 / self.s_w + (x - self.x_bar) ** 2 / self.sxx
        return math.sqrt(max(0.0, self.resid_var) * lever)


def weighted_line(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> Line:
    """Fit ``y = a + b x`` by weighted least squares."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return Line(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = y_bar - b * x_bar
    # Weighted residual variance, with two parameters spent.
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    resid_var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / max(1.0, len(xs)))
    return Line(a, b, x_bar, sxx, sw, resid_var)


# --- Per-trophy forecast ----------------------------------------------------
def _series(history: Sequence[dict[str, Any]], index: int, origin: str) -> list[tuple[float, float]]:
    """Return (hours_since_release, rate) pairs for trophy ``index``, nulls dropped."""
    points: list[tuple[float, float]] = []
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        points.append((hours_since(read["at"], origin), float(rate)))
    points.sort(key=lambda p: p[0])
    return points


def forecast_trophy(points: Sequence[tuple[float, float]], x_target: float, days: float) -> tuple[float, float]:
    """Return (median_rate, spread_in_logit_units) for one trophy.

    ``points`` are the trophy's reads; ``x_target`` the target's log-time coordinate;
    ``days`` the horizon in days (target minus checkpoint). Falls back to a
    persistence forecast when the history is too short to fit a trend.
    """
    base_spread = SIGMA_RW * math.sqrt(max(days, 0.0))
    if not points:
        return 0.0 + logit(0.0), base_spread + 0.20
    last_t, last_r = points[-1]
    last_l = logit(last_r)

    # Need at least three reads spanning a little time to trust a slope.
    span = last_t - points[0][0]
    if len(points) < 3 or span < 1.0:
        return last_l, base_spread + 0.20

    # Keep the most recent reads; weight the latest ones most strongly.
    recent = points[-60:]
    decay = math.log(2.0) / HALF_LIFE_HOURS
    xs = [math.log1p(t) for t, _ in recent]
    ys = [logit(r) for _, r in recent]
    ws = [math.exp(-decay * (last_t - t)) for t, _ in recent]

    line = weighted_line(xs, ys, ws)
    mu_fit = line.at(x_target)
    # Damp the extrapolated movement toward the last observation, and never
    # forecast a fall: an earn rate does not go down.
    mu = last_l + DAMPING * (mu_fit - last_l)
    mu = max(mu, last_l)

    # Spread: house random walk, plus the fit's own extrapolation error.
    s = math.sqrt(base_spread ** 2 + line.se_pred(x_target) ** 2)
    return mu, min(MAX_SPREAD, max(MIN_SPREAD, s))


# --- Case-level forecast ----------------------------------------------------
def forecast_case(case: dict[str, Any], name: str = "") -> dict[str, Any]:
    """Forecast one case into the contest's output object.

    Raises ``ValueError`` on a malformed case (the caller treats it as invalid).
    """
    if list(case.get("levels", LEVELS)) != list(LEVELS):
        raise ValueError(f"{name or case.get('id')}: unexpected levels {case.get('levels')}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name or case.get('id')}: empty history")

    origin = str(case.get("game", {}).get("release_date", "2026-11-19")) + "T00:00:00Z"
    target = case.get("target", TARGET_UTC)
    x_target = math.log1p(hours_since(target, origin))
    days = float(case.get("horizon_minutes", 0)) / 1440.0

    rows: list[dict[str, Any]] = []
    for index, trophy in enumerate(case["trophies"]):
        points = _series(history, index, origin)
        mu, s = forecast_trophy(points, x_target, days)
        q = [round4(inv_logit(mu + z * s)) for z in Z_LEVELS]
        # Guarantee the non-decreasing quantile contract.
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def last_rates(case: dict[str, Any]) -> dict[int, float]:
    """The rate of every trophy in the case's last read (the baseline), by id."""
    last = case["history"][-1]
    out: dict[int, float] = {}
    for index, trophy in enumerate(case["trophies"]):
        rates = last.get("rates") or []
        rate = rates[index] if index < len(rates) else None
        out[trophy["id"]] = float(rate) if rate is not None else 0.0
    return out


def enforce_monotonicity(rows: Iterable[dict[str, Any]]) -> None:
    """Platinum (id 1) must not exceed the rarest base-game median (in place).

    A cheap sanity guard on the medians only; it never reorders quantiles.
    """
    rows = list(rows)
    if not rows:
        return
    platinum = next((r for r in rows if r["id"] == 1), None)
    if platinum is None:
        return
    others = [r["q"][2] for r in rows if r["id"] != 1]
    if others:
        cap = min(others)
        if platinum["q"][2] > cap:
            shift = platinum["q"][2] - cap
            platinum["q"] = [max(0.0, v - shift) for v in platinum["q"]]


def validate(case: dict[str, Any], forecast: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty when the forecast satisfies the contract)."""
    problems: list[str] = []
    trophies = case["trophies"]
    rows = forecast.get("trophies", [])
    seen = {r["id"] for r in rows}
    if len(rows) != len(trophies):
        problems.append(f"row count {len(rows)} != trophy count {len(trophies)}")
    if seen != {t["id"] for t in trophies}:
        problems.append("trophy ids are not matched one for one")
    for row in rows:
        q = row.get("q")
        if not isinstance(q, list) or len(q) != len(LEVELS):
            problems.append(f"trophy {row.get('id')}: q is not {len(LEVELS)} numbers")
            continue
        for value in q:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not (0.0 <= value <= 100.0):
                problems.append(f"trophy {row['id']}: quantile out of (0,100)")
                break
        if any(q[i] < q[i - 1] for i in range(1, len(q))):
            problems.append(f"trophy {row['id']}: quantiles not non-decreasing")
    if json.dumps(forecast, separators=(",", ":")).__len__() > 65536:
        problems.append("output exceeds 65536 bytes")
    return problems
