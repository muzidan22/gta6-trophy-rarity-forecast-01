"""Offline verification for the forecaster: a synthetic full-week case + score.

The kit only ships launch-night practice cases (p00/p01), whose target is a page
refresh 90 minutes on; the final case f00 is cut at the deadline and held by the
grade. This script builds a synthetic week-long case from the kit's real 55 trophies
so the parser, the trend model, the cutoff filter and the quantile contract can be
exercised and scored end to end, then writes the ``output/`` artifacts.

    python3 verify_mock.py --kit-dir <unpacked kit> --output-dir output
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecaster as fc  # noqa: E402
import main as app  # noqa: E402

KIT_DEFAULT = Path(__file__).resolve().parent.parent / "gta6-trophy-rarity-forecast-01-kit"


def synth_case(kit_dir: Path, seed: int = 7) -> tuple[dict, list[float]]:
    """Build a 7-read history over 59.5h from the kit's trophies and a truth at target."""
    kit_case = json.loads((kit_dir / "input" / "p00.in").read_text(encoding="utf-8"))
    trophies = kit_case["trophies"]
    rng = random.Random(seed)
    reads: list[dict] = []
    # seven reads from launch night to the cutoff, two days in
    for hour in (0.5, 6, 12, 24, 36, 48, 59.5):
        rates = []
        for trophy in trophies:
            # a saturating growth curve in logit space, faster for easy tiers
            tier = trophy["tier"]
            rate_at_week = {"platinum": 0.9, "gold": 6.0, "silver": 22.0, "bronze": 55.0}[tier]
            ceiling = fc.logit(rate_at_week)
            floor = fc.logit(0.3)
            grow = max(0.0, ceiling - floor) * (1.0 - math.exp(-hour / 40.0))
            mu = floor + grow + rng.gauss(0.0, 0.03)
            rates.append(round(fc.inv_logit(mu), 2))
        reads.append({"slot": f"synth-{hour}", "at": _iso(hour), "players_tracked": 638,
                      "trophies_earned": None, "rates": rates})
    truth = list(reads[-1]["rates"])
    # the answering read a week out: the same curve at 132h
    for i, trophy in enumerate(trophies):
        tier = trophy["tier"]
        rate_at_week = {"platinum": 0.9, "gold": 6.0, "silver": 22.0, "bronze": 55.0}[tier]
        ceiling = fc.logit(rate_at_week)
        floor = fc.logit(0.3)
        grow = max(0.0, ceiling - floor) * (1.0 - math.exp(-132.0 / 40.0))
        truth[i] = round(fc.inv_logit(floor + grow + rng.gauss(0.0, 0.02)), 2)
    case = app.build_case(trophies, reads, case_id="mock", at=reads[-1]["at"], target=fc.TARGET_UTC)
    return case, truth


def _iso(hours: float) -> str:
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    return (base + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def pinball(quantiles: list[float], truth: float) -> float:
    """The contest's mean pinball loss on the logit scale, one trophy (5 items)."""
    y = fc.logit(truth)
    total = 0.0
    for tau, q in zip(fc.LEVELS, quantiles):
        d = y - fc.logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(fc.LEVELS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-dir", default=str(KIT_DEFAULT))
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    case, truth = synth_case(Path(args.kit_dir))
    forecast = fc.forecast_case(case, "mock")
    fc.enforce_monotonicity(forecast["trophies"])

    problems = fc.validate(case, forecast)
    assert not problems, problems
    assert len(forecast["trophies"]) == 55, "expected all 55 trophies"

    cut = fc.parse_utc(fc.CUTOFF_UTC)
    assert all(fc.parse_utc(h["at"]) <= cut for h in case["history"]), "post-cutoff leak"
    by_id = {r["id"]: r for r in forecast["trophies"]}
    baseline = fc.last_rates(case)
    losses = [pinball(by_id[t["id"]]["q"], truth[i]) for i, t in enumerate(case["trophies"])]
    base_losses = [pinball([baseline[t["id"]]] * 5, truth[i]) for i, t in enumerate(case["trophies"])]
    ours = sum(losses) / len(losses)
    base = sum(base_losses) / len(base_losses)
    covered = sum(1 for i, t in enumerate(case["trophies"])
                  if by_id[t["id"]]["q"][0] <= truth[i] <= by_id[t["id"]]["q"][4]) / len(case["trophies"])

    rows = app.write_artifacts(case, Path(args.output_dir))
    print(f"case {case['id']}: {len(rows)} trophies, cutoff {fc.CUTOFF_UTC} respected")
    print(f"mock loss  model={ours:.4f}  persistence-baseline={base:.4f}  90% coverage={covered:.2f}")
    assert ours < base, "the trend model must beat persistence on the synthetic week"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
