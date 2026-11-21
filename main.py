"""Entrypoint for make.fun's gta6-trophy-rarity-forecast-01 (program/v1).

Two modes, chosen by the arguments:

* Program mode (the grader's): with no CLI flags it reads every ``*.in`` case in the
  input directory (``$MAKEFUN_INPUT_DIR`` or ``/input``), oldest-first histories of
  the game's Exophase reads, and writes one forecast per case to
  ``$MAKEFUN_OUTPUT_DIR`` (or ``/output``) as ``<id>.out``. Two runs on the same case
  write the same bytes (nothing is random, files are processed in name order).

* Pipeline mode (offline, reproducible): ``--input-html`` parses a saved Exophase
  snapshot, ``--history-csv`` ingests a logged time series, ``--live`` fetches the
  page (curl_cffi, Chrome TLS impersonation, with a UTC stamp) and ``--output-dir``
  writes ``gta6_trophy_forecast.csv`` and ``.json`` for all of the game's trophies.

Run it the way the grader does::

    python3 -P -s main.py

Run the pipeline::

    python3 main.py --history-csv history.csv --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The grader's -P keeps this file's directory OFF sys.path: put it back so that
# ``import forecaster`` and ``import scraper`` resolve to the files beside main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecaster as fc  # noqa: E402
import scraper as sc  # noqa: E402

CUTOFF_UTC = fc.CUTOFF_UTC
TARGET_UTC = fc.TARGET_UTC


def now_utc() -> str:
    """The current UTC time as an ISO 8601 string with a Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Program mode (the grader) ---------------------------------------------
def run_program(input_dir: Path, output_dir: Path) -> int:
    """Forecast every ``*.in`` case in ``input_dir`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(input_dir.glob("*.in"))
    if not cases:
        print(f"no *.in case files in {input_dir}", file=sys.stderr)
        return 1
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = fc.forecast_case(case, path.name)
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    return 0


# --- Pipeline mode ----------------------------------------------------------
def _read_history_csv(path: Path) -> list[dict[str, Any]]:
    """Read a read-log CSV into history rows.

    The first column must be ``at`` (ISO 8601 UTC) or ``slot``; a ``players_tracked``
    column is optional; every other column named by an integer is a trophy rate.
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        time_col = "at" if "at" in reader.fieldnames else ("slot" if "slot" in reader.fieldnames else reader.fieldnames[0])
        trophy_cols = [f for f in reader.fieldnames if f not in (time_col, "players_tracked", "trophies_earned")]
        for raw in reader:
            rates = []
            for col in trophy_cols:
                value = (raw.get(col) or "").strip()
                rates.append(float(value) if value not in ("", "NA", "null") else None)
            rows.append({
                "slot": raw.get("slot", raw.get(time_col, "")),
                "at": raw.get(time_col, ""),
                "players_tracked": int(raw["players_tracked"]) if raw.get("players_tracked") else None,
                "trophies_earned": int(raw["trophies_earned"]) if raw.get("trophies_earned") else None,
                "rates": rates,
            })
    rows.sort(key=lambda r: r["at"])
    return rows


def build_case(trophies: list[dict[str, Any]], history: list[dict[str, Any]], case_id: str,
               at: str, target: str) -> dict[str, Any]:
    """Assemble a case object of the kit's schema from trophies and a history."""
    horizon = (fc.parse_utc(target) - fc.parse_utc(at)).total_seconds() / 60.0
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": case_id,
        "at": at,
        "horizon_minutes": round(horizon),
        "target": target,
        "levels": list(fc.LEVELS),
        "game": {
            "name": "Grand Theft Auto VI",
            "platform": "PS5",
            "release_date": "2026-11-19",
            "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/",
        },
        "trophies": trophies,
        "history": [h for h in history if fc.parse_utc(h["at"]) <= fc.parse_utc(CUTOFF_UTC)],
        "notes": "Assembled offline by main.py; filtered to the strict cutoff " + CUTOFF_UTC + ".",
    }


def _enforce_cutoff(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop every read at or after the strict cutoff (never leak post-cutoff data)."""
    cutoff = fc.parse_utc(CUTOFF_UTC)
    return [h for h in history if fc.parse_utc(h["at"]) <= cutoff]


def write_artifacts(case: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Write the CSV/JSON forecast artifacts and return the rows."""
    forecast = fc.forecast_case(case, case.get("id", "case"))
    fc.enforce_monotonicity(forecast["trophies"])
    by_id = {row["id"]: row for row in forecast["trophies"]}
    baseline = fc.last_rates(case)
    timestamps = {h["at"] for h in case["history"]}
    mode = "A" if len(timestamps) >= 2 else "B"

    rows: list[dict[str, Any]] = []
    for trophy in case["trophies"]:
        row = by_id[trophy["id"]]
        base = round(baseline.get(trophy["id"], 0.0), 2)
        median = round(float(row["q"][2]), 2)
        rows.append({
            "trophy_id": trophy["id"],
            "title": trophy["name"],
            "cutoff_timestamp_utc": CUTOFF_UTC,
            "baseline_earn_rate_pct": base,
            "forecast_timestamp_utc": TARGET_UTC,
            "forecast_earn_rate_pct": median,
            "delta_pct": round(median - base, 2),
            "model_mode_used": mode,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gta6_trophy_forecast.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "competition": "gta6-trophy-rarity-forecast-01",
        "cutoff_timestamp_utc": CUTOFF_UTC,
        "forecast_timestamp_utc": TARGET_UTC,
        "model_mode_used": mode,
        "generated_at_utc": now_utc(),
        "trophy_count": len(rows),
        "forecast": rows,
        "quantiles_by_trophy": [
            {"trophy_id": t["id"], "q05": forecast["trophies"][i]["q"][0], "q25": forecast["trophies"][i]["q"][1],
             "q50": forecast["trophies"][i]["q"][2], "q75": forecast["trophies"][i]["q"][3],
             "q95": forecast["trophies"][i]["q"][4]}
            for i, t in enumerate(case["trophies"])
        ],
    }
    (out_dir / "gta6_trophy_forecast.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return rows


def run_pipeline(args: argparse.Namespace) -> int:
    """Offline pipeline: ingest a snapshot/history and write the artifacts."""
    out_dir = Path(args.output_dir or "output")
    history: list[dict[str, Any]] = []
    trophies: list[dict[str, Any]] = []

    if args.history_csv:
        history = _read_history_csv(Path(args.history_csv))
        trophy_ids = sorted({int(c) for c in csv.DictReader(Path(args.history_csv).open(encoding="utf-8")).fieldnames
                             if str(c).isdigit()})
        trophies = [{"id": i, "name": f"Trophy {i}", "description": "", "tier": "", "points": 0, "secret": False}
                    for i in trophy_ids]
    elif args.input_html:
        snapshot = sc.parse_html(Path(args.input_html).read_text(encoding="utf-8", errors="replace"),
                                 timestamp_utc=args.timestamp or now_utc())
        trophies = [{"id": t.trophy_id, "name": t.title, "description": t.description, "tier": t.grade,
                     "points": t.points, "secret": False} for t in snapshot.trophies]
        history = [{"slot": snapshot.timestamp_utc, "at": snapshot.timestamp_utc, "players_tracked": None,
                    "trophies_earned": None, "rates": [t.earn_rate_pct for t in snapshot.trophies]}]
    elif args.live:
        snapshot = sc.fetch_live()
        trophies = [{"id": t.trophy_id, "name": t.title, "description": t.description, "tier": t.grade,
                     "points": t.points, "secret": False} for t in snapshot.trophies]
        history = [{"slot": snapshot.timestamp_utc, "at": snapshot.timestamp_utc, "players_tracked": None,
                    "trophies_earned": None, "rates": [t.earn_rate_pct for t in snapshot.trophies]}]
    else:
        print("pipeline mode needs --history-csv, --input-html or --live", file=sys.stderr)
        return 2

    if not history:
        print("no reads ingested", file=sys.stderr)
        return 1

    history = _enforce_cutoff(history)
    if not history:
        print(f"every read is after the cutoff {CUTOFF_UTC}", file=sys.stderr)
        return 1

    case = build_case(trophies, history, case_id="snapshot", at=history[-1]["at"], target=TARGET_UTC)
    problems = fc.validate(case, fc.forecast_case(case, "snapshot"))
    if problems:
        print("validation problems: " + "; ".join(problems), file=sys.stderr)
        return 1
    rows = write_artifacts(case, out_dir)
    print(f"wrote {len(rows)} trophy forecasts to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GTA VI trophy rarity forecaster")
    parser.add_argument("--input-html", help="a saved Exophase trophy-page HTML snapshot")
    parser.add_argument("--history-csv", help="a logged time series of reads (CSV)")
    parser.add_argument("--live", action="store_true", help="fetch the page live (curl_cffi)")
    parser.add_argument("--output-dir", help="where the CSV/JSON artifacts go")
    parser.add_argument("--timestamp", help="UTC stamp to attach to a local snapshot")
    args = parser.parse_args(argv)

    if args.input_html or args.history_csv or args.live or args.output_dir:
        return run_pipeline(args)

    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    return run_program(input_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
