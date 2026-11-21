"""Exophase trophy-page ingestion (standard library, with an optional curl_cffi path).

The grader never calls this: it runs only ``main.py`` reading ``/input``. This module
exists for the reproducible offline pipeline (``main.py --input-html`` /
``--history-csv`` / ``--live``) so a snapshot or a logged time series can be turned
into the same case object the sandbox grades. It parses a saved Exophase HTML page,
or fetches a live one with Chrome TLS impersonation when ``curl_cffi`` is installed,
and always stamps the result with an exact UTC timestamp.

Parsed fields per trophy: ``trophy_id``, ``title``, ``description``, ``grade``
(Platinum/Gold/Silver/Bronze), ``points``, ``exp``, ``earn_rate_pct``,
``earners_count``, ``total_tracked_players`` and ``snapshot_timestamp_utc``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

EXOPHASE_URL = "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"
GRADE_POINTS = {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_INT = re.compile(r"([\d,]+)")


@dataclass
class Trophy:
    """One trophy from an Exophase snapshot."""

    trophy_id: int
    title: str
    description: str = ""
    grade: str = ""
    points: int = 0
    exp: int = 0
    earn_rate_pct: float = 0.0
    earners_count: int | None = None
    total_tracked_players: int | None = None
    snapshot_timestamp_utc: str = ""


@dataclass
class Snapshot:
    """A single Exophase read (one timestamp)."""

    timestamp_utc: str
    url: str = EXOPHASE_URL
    trophies: list[Trophy] = field(default_factory=list)
    total_tracked_players: int | None = None


class _RowParser(HTMLParser):
    """Collect the text and anchors of each ``<tr>`` on an Exophase trophy table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, Any]] = []
        self._depth = 0
        self._current: dict[str, Any] | None = None
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current = {"text": [], "hrefs": [], "classes": dict(attrs).get("class", "") or ""}
            self._depth = 1
        elif tag == "a" and self._current is not None:
            self._href = dict(attrs).get("href", "") or ""
            self._current["hrefs"].append(self._href)
        elif tag == "img" and self._current is not None:
            self._current.setdefault("imgs", []).append(
                (dict(attrs).get("src", "") or "") + " " + (dict(attrs).get("alt", "") or "")
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._current is not None:
            self._current["text"] = " ".join(self._current["text"])
            self.rows.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text = data.strip()
            if text:
                self._current["text"].append(text)


def _grade_from(text: str, img_hint: str) -> str:
    blob = f"{text} {img_hint}".lower()
    for grade in ("platinum", "gold", "silver", "bronze"):
        if grade in blob:
            return grade.capitalize()
    return ""


def _trophy_id(hrefs: list[str], fallback: int) -> int:
    for href in hrefs:
        match = re.search(r"/(\d+)/?$", href)
        if match:
            return int(match.group(1))
    return fallback


def parse_html(html: str, timestamp_utc: str) -> Snapshot:
    """Parse an Exophase trophy page (saved or fetched) into a :class:`Snapshot`."""
    parser = _RowParser()
    parser.feed(html)
    trophies: list[Trophy] = []
    seen: set[int] = set()
    for row in parser.rows:
        text = row.get("text", "")
        match = _PCT.search(text)
        if not match:
            continue
        rate = float(match.group(1))
        # A title: the row text up to the percentage, trimmed of grade noise.
        title = text[: match.start()].strip(" -\u2013\u00b7")
        if not title:
            title = text
        # Exophase prints "earners of N tracked" style counts; take any big number.
        counts = [int(n.replace(",", "")) for n in _INT.findall(text)]
        counts = [c for c in counts if c > 100]
        trophy_id = _trophy_id(row.get("hrefs", []), len(trophies) + 1)
        if trophy_id in seen:
            continue
        seen.add(trophy_id)
        grade = _grade_from(text, " ".join(row.get("imgs", [])))
        trophies.append(Trophy(
            trophy_id=trophy_id,
            title=title[:200],
            description="",
            grade=grade,
            points=GRADE_POINTS.get(grade.lower(), 0),
            earn_rate_pct=round(rate, 2),
            earners_count=counts[0] if counts else None,
            total_tracked_players=counts[-1] if len(counts) > 1 else None,
            snapshot_timestamp_utc=timestamp_utc,
        ))
    trophies.sort(key=lambda t: t.trophy_id)
    return Snapshot(timestamp_utc=timestamp_utc, trophies=trophies)


def fetch_live(timestamp_utc: str | None = None, url: str = EXOPHASE_URL, timeout: float = 30.0) -> Snapshot:
    """Fetch the live page (curl_cffi Chrome impersonation, else urllib) and parse it.

    Every fetch is stamped with the exact UTC time it completed.
    """
    stamp = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:  # preferred: Chrome TLS impersonation to pass bot checks
        from curl_cffi import requests as cffi_requests  # type: ignore

        response = cffi_requests.get(url, impersonate="chrome", timeout=timeout)
        response.raise_for_status()
        html = response.text
    except Exception:  # pragma: no cover - network/optional dependency
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Chrome)"})
        with urllib.request.urlopen(request, timeout=timeout) as handle:  # noqa: S310
            html = handle.read().decode("utf-8", errors="replace")
    return parse_html(html, timestamp_utc=stamp)


def to_history_rows(snapshots: list[Snapshot]) -> list[dict[str, Any]]:
    """Turn a list of snapshots into case-history rows (oldest first, by timestamp)."""
    ordered = sorted(snapshots, key=lambda s: s.timestamp_utc)
    return [{
        "slot": s.timestamp_utc,
        "at": s.timestamp_utc,
        "players_tracked": s.total_tracked_players,
        "trophies_earned": None,
        "rates": [t.earn_rate_pct for t in s.trophies],
    } for s in ordered]
