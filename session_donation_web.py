#!/usr/bin/env python3
"""
Local web UI for project-scoped Codex and Claude Code session donation.

The server binds to localhost, scans local session projects, estimates donation
volume, packages selected sessions, and reuses donate_project_sessions.py for
privacy filtering and verification.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import donate_project_sessions as donation


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TOKEN_CHAR_RATIO = 4
SCAN_CACHE_TTL_SECONDS = 300


@dataclass
class SessionStats:
    characters: int
    bytes: int
    estimated_tokens: int
    messages: int
    tool_events: int


@dataclass
class IndexedSession:
    candidate: donation.SessionCandidate
    project_root: Path
    stats: SessionStats


SCAN_CACHE: dict[str, Any] = {
    "key": None,
    "created_at": 0.0,
    "projects": [],
    "sessions": {},
    "totals": {},
    "excluded": {},
}


def parse_sources(raw_sources: str) -> set[str]:
    return donation.parse_sources(raw_sources)


def find_git_root(path: Path) -> Path | None:
    current = donation.canonical_path(path)
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def infer_project_root(cwd_values: list[str]) -> Path | None:
    if not cwd_values:
        return None

    cwd_paths = [donation.canonical_path(Path(cwd)) for cwd in cwd_values]
    git_roots = [find_git_root(path) for path in cwd_paths]
    if all(root is not None for root in git_roots):
        first = git_roots[0]
        if all(root == first for root in git_roots):
            return first

    first_cwd = cwd_paths[0]
    if all(path == first_cwd or donation.is_relative_to(path, first_cwd) for path in cwd_paths):
        return first_cwd

    return None


def session_stats(parsed: donation.ParsedSession, include_tools: bool) -> SessionStats:
    texts = [message.content for message in parsed.messages]
    if include_tools:
        texts.extend(event.content for event in parsed.tool_events)

    characters = sum(len(text) for text in texts)
    byte_count = sum(len(text.encode("utf-8", errors="replace")) for text in texts)
    estimated_tokens = math.ceil(characters / TOKEN_CHAR_RATIO) if characters else 0
    return SessionStats(
        characters=characters,
        bytes=byte_count,
        estimated_tokens=estimated_tokens,
        messages=len(parsed.messages),
        tool_events=len(parsed.tool_events) if include_tools else 0,
    )


def merge_stats(total: dict[str, int], stats: SessionStats) -> None:
    total["characters"] = total.get("characters", 0) + stats.characters
    total["bytes"] = total.get("bytes", 0) + stats.bytes
    total["estimated_tokens"] = total.get("estimated_tokens", 0) + stats.estimated_tokens
    total["messages"] = total.get("messages", 0) + stats.messages
    total["tool_events"] = total.get("tool_events", 0) + stats.tool_events


def empty_totals() -> dict[str, int]:
    return {
        "characters": 0,
        "bytes": 0,
        "estimated_tokens": 0,
        "messages": 0,
        "tool_events": 0,
        "sessions": 0,
    }


def iter_parsed_sessions(sources: set[str], include_tools: bool) -> list[tuple[str, Path, donation.ParsedSession | None]]:
    parsed_sessions: list[tuple[str, Path, donation.ParsedSession | None]] = []

    if "codex" in sources:
        for installation in donation.codex_installations():
            for path in donation.discover_codex_session_files(installation):
                parsed_sessions.append(("codex", path, donation.parse_codex_session(path, include_tools=include_tools)))

    if "claude" in sources or "claude-code" in sources:
        for installation in donation.claude_installations():
            for path in donation.discover_claude_session_files(installation):
                parsed_sessions.append(
                    ("claude-code", path, donation.parse_claude_session(path, include_tools=include_tools))
                )

    return parsed_sessions


def build_project_index(sources: set[str], include_tools: bool) -> dict[str, Any]:
    started = time.time()
    sessions: dict[str, IndexedSession] = {}
    projects_by_root: dict[str, dict[str, Any]] = {}
    totals = empty_totals()
    excluded: dict[str, int] = {}

    for source, source_path, parsed in iter_parsed_sessions(sources, include_tools):
        if parsed is None:
            excluded["no exportable user or assistant messages"] = (
                excluded.get("no exportable user or assistant messages", 0) + 1
            )
            continue

        project_root = infer_project_root(parsed.cwd_values)
        if project_root is None:
            excluded["mixed or missing project cwd"] = excluded.get("mixed or missing project cwd", 0) + 1
            continue

        eligible, reason, cwds = donation.session_eligibility(parsed, project_root)
        if not eligible:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue

        candidate = donation.SessionCandidate(
            parsed=parsed,
            source=source,
            source_path=source_path,
            eligible=True,
            reason="eligible",
            cwd_values=cwds,
        )
        session_id = donation.candidate_hash(candidate)
        stats = session_stats(parsed, include_tools)
        indexed = IndexedSession(candidate=candidate, project_root=project_root, stats=stats)
        sessions[session_id] = indexed

        root_key = str(project_root)
        project = projects_by_root.setdefault(
            root_key,
            {
                "id": donation.stable_hash(root_key)[:16],
                "name": project_root.name or root_key,
                "path": root_key,
                "sources": set(),
                "stats": empty_totals(),
                "sessions": [],
                "latest_updated_at": None,
            },
        )
        project["sources"].add(parsed.source)
        project["sessions"].append(session_id)
        project["stats"]["sessions"] += 1
        merge_stats(project["stats"], stats)
        totals["sessions"] += 1
        merge_stats(totals, stats)

        updated = parsed.updated_at or parsed.created_at
        if updated and (project["latest_updated_at"] is None or updated > project["latest_updated_at"]):
            project["latest_updated_at"] = updated

    projects = []
    for project in projects_by_root.values():
        project_sessions = [session_payload(session_id, sessions[session_id]) for session_id in project["sessions"]]
        project_sessions.sort(
            key=lambda item: (
                item["stats"].get("estimated_tokens", 0),
                item["stats"].get("bytes", 0),
                item.get("updated_at") or "",
            ),
            reverse=True,
        )
        project["sessions"] = project_sessions
        project["sources"] = sorted(project["sources"])
        projects.append(project)

    projects.sort(
        key=lambda project: (
            project["stats"].get("estimated_tokens", 0),
            project["stats"].get("bytes", 0),
            project["stats"].get("sessions", 0),
        ),
        reverse=True,
    )

    return {
        "scanned_at": donation.utc_now_iso(),
        "scan_seconds": round(time.time() - started, 3),
        "include_tools": include_tools,
        "sources": sorted(sources),
        "projects": projects,
        "sessions": sessions,
        "totals": totals,
        "excluded": excluded,
    }


def session_payload(session_id: str, indexed: IndexedSession) -> dict[str, Any]:
    parsed = indexed.candidate.parsed
    assert parsed is not None
    return {
        "id": session_id,
        "source": parsed.source,
        "created_at": parsed.created_at,
        "updated_at": parsed.updated_at,
        "stats": stats_payload(indexed.stats),
    }


def stats_payload(stats: SessionStats | dict[str, int]) -> dict[str, int]:
    if isinstance(stats, SessionStats):
        return {
            "characters": stats.characters,
            "bytes": stats.bytes,
            "estimated_tokens": stats.estimated_tokens,
            "messages": stats.messages,
            "tool_events": stats.tool_events,
        }
    return dict(stats)


def cached_index(sources: set[str], include_tools: bool, refresh: bool = False) -> dict[str, Any]:
    key = json.dumps({"sources": sorted(sources), "include_tools": include_tools}, sort_keys=True)
    now = time.time()
    if (
        not refresh
        and SCAN_CACHE["key"] == key
        and SCAN_CACHE["created_at"]
        and now - SCAN_CACHE["created_at"] < SCAN_CACHE_TTL_SECONDS
    ):
        return SCAN_CACHE

    index = build_project_index(sources, include_tools)
    SCAN_CACHE.clear()
    SCAN_CACHE.update(index)
    SCAN_CACHE["key"] = key
    SCAN_CACHE["created_at"] = now
    return SCAN_CACHE


def public_index(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "scanned_at": index["scanned_at"],
        "scan_seconds": index["scan_seconds"],
        "include_tools": index["include_tools"],
        "sources": index["sources"],
        "projects": index["projects"],
        "totals": index["totals"],
        "excluded": index["excluded"],
        "token_estimate": {
            "method": f"ceil(characters / {TOKEN_CHAR_RATIO})",
            "note": "Tokenizer-free local estimate for planning donation volume.",
        },
    }


def selected_sessions_from_payload(payload: dict[str, Any], index: dict[str, Any]) -> list[IndexedSession]:
    selected_ids = {str(item) for item in payload.get("session_ids", [])}
    selected_project_ids = {str(item) for item in payload.get("project_ids", [])}

    for project in index["projects"]:
        if project["id"] in selected_project_ids:
            selected_ids.update(session["id"] for session in project["sessions"])

    selected: list[IndexedSession] = []
    sessions: dict[str, IndexedSession] = index["sessions"]
    for session_id in sorted(selected_ids):
        indexed = sessions.get(session_id)
        if indexed:
            selected.append(indexed)
    return selected


def package_selected_sessions(
    selected: list[IndexedSession],
    output_root: Path,
    privacy_filter_command: str | None,
    include_tools: bool,
) -> dict[str, Any]:
    if not selected:
        raise ValueError("Select at least one session or project before packaging.")

    privacy_filter = donation.PrivacyFilter(privacy_filter_command)
    privacy_filter.ensure_available()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = donation.canonical_path(output_root / f"web_package_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    project_stats: dict[str, dict[str, Any]] = {}
    totals = empty_totals()

    for indexed in selected:
        candidate = indexed.candidate
        project_root = indexed.project_root
        project_key = str(project_root)
        project_hash = donation.stable_hash(project_key)
        minimizer = donation.LocalMinimizer(project_root)
        records.append(
            donation.export_session(
                candidate=candidate,
                project_hash=project_hash,
                minimizer=minimizer,
                privacy_filter=privacy_filter,
                include_tools=include_tools,
            )
        )

        project = project_stats.setdefault(
            project_key,
            {
                "root_hash": project_hash,
                "name_hint": project_root.name,
                "session_count": 0,
                "stats": empty_totals(),
            },
        )
        project["session_count"] += 1
        project["stats"]["sessions"] += 1
        merge_stats(project["stats"], indexed.stats)
        totals["sessions"] += 1
        merge_stats(totals, indexed.stats)

    donation_path = output_dir / donation.SHAREABLE_DONATION_FILE
    with donation_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": donation.SCHEMA_VERSION,
        "created_at": donation.utc_now_iso(),
        "project": {
            "root_hash": donation.stable_hash("|".join(sorted(project_stats))),
            "name_hint": "web-selection" if len(project_stats) != 1 else next(iter(project_stats.values()))["name_hint"],
        },
        "projects": [
            {
                "root_hash": project["root_hash"],
                "name_hint": project["name_hint"],
                "session_count": project["session_count"],
                "stats": project["stats"],
            }
            for project in project_stats.values()
        ],
        "privacy_filter": {
            "model": donation.PRIVACY_FILTER_MODEL,
            "runner": privacy_filter.runner,
            "status": "filtered",
        },
        "options": {
            "include_tools": include_tools,
            "interface": "local-web",
            "token_estimate": f"ceil(characters / {TOKEN_CHAR_RATIO})",
        },
        "counts": {
            "candidates": len(selected),
            "eligible": len(selected),
            "donated": len(records),
            "excluded": 0,
        },
        "stats": totals,
        "files": {
            "donation_jsonl": donation.SHAREABLE_DONATION_FILE,
            "donation_sha256": donation.file_sha256(donation_path),
            "review_html": donation.REVIEW_FILE,
        },
        "donated_sessions": [
            {
                "source": indexed.candidate.source,
                "session_hash": donation.candidate_hash(indexed.candidate),
                "message_count": indexed.stats.messages,
                "tool_event_count": indexed.stats.tool_events,
                "estimated_tokens": indexed.stats.estimated_tokens,
                "created_at": indexed.candidate.parsed.created_at if indexed.candidate.parsed else None,
                "updated_at": indexed.candidate.parsed.updated_at if indexed.candidate.parsed else None,
            }
            for indexed in selected
        ],
        "excluded_summary": {},
        "verify_command": "python3 donate_project_sessions.py verify <output-directory>",
    }

    manifest_path = output_dir / donation.SHAREABLE_MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    review_path = output_dir / donation.REVIEW_FILE
    review_path.write_text(donation.render_review_html(manifest, records), encoding="utf-8")

    zip_base = output_dir
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_dir))

    return {
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "review_url": f"/review/{output_dir.name}",
        "download_url": f"/download/{zip_path.name}",
        "manifest": manifest,
    }


def verify_output_dir(output_dir: str) -> dict[str, Any]:
    buffer = io.StringIO()
    args = argparse.Namespace(output_dir=output_dir)
    with contextlib.redirect_stdout(buffer):
        code = donation.command_verify(args)
    return {
        "ok": code == 0,
        "exit_code": code,
        "output": buffer.getvalue(),
    }


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/html") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("JSON request body must be an object")
    return parsed


class DonationWebHandler(BaseHTTPRequestHandler):
    server_version = "SessionDonationWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            text_response(self, HTML_PAGE)
            return

        if parsed.path == "/api/scan":
            query = parse_qs(parsed.query)
            sources = parse_sources(query.get("sources", ["codex,claude"])[0])
            include_tools = query.get("include_tools", ["0"])[0] == "1"
            refresh = query.get("refresh", ["0"])[0] == "1"
            try:
                index = cached_index(sources, include_tools, refresh=refresh)
                json_response(self, {"ok": True, **public_index(index)})
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, status=500)
            return

        if parsed.path.startswith("/review/"):
            package_name = safe_name(parsed.path.removeprefix("/review/"))
            review_path = donation.canonical_path(Path(donation.DEFAULT_OUTPUT_ROOT) / package_name / donation.REVIEW_FILE)
            if review_path.exists():
                text_response(self, review_path.read_text(encoding="utf-8"))
            else:
                text_response(self, "Review not found", status=404, content_type="text/plain")
            return

        if parsed.path.startswith("/download/"):
            zip_name = safe_name(parsed.path.removeprefix("/download/"))
            zip_path = donation.canonical_path(Path(donation.DEFAULT_OUTPUT_ROOT) / zip_name)
            if zip_path.exists() and zip_path.suffix == ".zip":
                body = zip_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{zip_path.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                text_response(self, "Package not found", status=404, content_type="text/plain")
            return

        text_response(self, "Not found", status=404, content_type="text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/package":
            try:
                payload = read_request_json(self)
                sources = parse_sources(str(payload.get("sources", "codex,claude")))
                include_tools = bool(payload.get("include_tools", False))
                index = cached_index(sources, include_tools, refresh=bool(payload.get("refresh", False)))
                selected = selected_sessions_from_payload(payload, index)
                result = package_selected_sessions(
                    selected=selected,
                    output_root=Path(donation.DEFAULT_OUTPUT_ROOT),
                    privacy_filter_command=payload.get("privacy_filter_command") or None,
                    include_tools=include_tools,
                )
                json_response(self, {"ok": True, **result})
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, status=400)
            return

        if parsed.path == "/api/verify":
            try:
                payload = read_request_json(self)
                result = verify_output_dir(str(payload.get("output_dir", "")))
                json_response(self, result, status=200 if result["ok"] else 400)
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, status=400)
            return

        text_response(self, "Not found", status=404, content_type="text/plain")


def safe_name(value: str) -> str:
    decoded = unquote(value)
    name = Path(decoded).name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return ""
    return name


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Session Donation Packager</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --ink: #15202b;
      --muted: #5d6b7a;
      --line: #d9e0ea;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
      --soft: #eef6f5;
      --shadow: 0 12px 32px rgba(21, 32, 43, .07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.4;
    }
    main {
      width: min(1240px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }
    header.top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 1.7rem;
      letter-spacing: 0;
    }
    p { margin: 0; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: start;
    }
    .panel, .project {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 14px; }
    .controls {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) minmax(260px, 2fr) max-content;
      gap: 10px;
      margin-bottom: 14px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: .82rem;
      font-weight: 650;
    }
    input[type="text"], input[type="search"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }
    .checkline {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--ink);
      font-size: .92rem;
      font-weight: 500;
      min-height: 38px;
      padding-top: 20px;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 11px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.primary:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: not-allowed;
      opacity: .55;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin: 8px 0 14px;
    }
    .toolbar .left, .toolbar .right {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .project {
      margin-bottom: 10px;
      overflow: hidden;
    }
    .project-head {
      display: grid;
      grid-template-columns: auto auto minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
    }
    .disclosure {
      width: 30px;
      height: 30px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      font-size: 1rem;
      line-height: 1;
    }
    .project-title {
      min-width: 0;
    }
    .project-title strong {
      display: block;
      overflow-wrap: anywhere;
      font-size: .98rem;
    }
    .project-title small {
      display: block;
      color: var(--muted);
      overflow-wrap: anywhere;
      margin-top: 2px;
    }
    .badges {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .badge {
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      font-size: .78rem;
      white-space: nowrap;
    }
    .session-list {
      border-top: 1px solid var(--line);
      background: #fbfcfe;
      padding: 6px 14px 10px;
    }
    .session-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid #edf1f6;
    }
    .session-row:last-child { border-bottom: 0; }
    .session-main {
      min-width: 0;
      color: var(--muted);
      font-size: .84rem;
      overflow-wrap: anywhere;
    }
    aside {
      position: sticky;
      top: 16px;
      display: grid;
      gap: 12px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: .78rem;
    }
    .metric strong {
      display: block;
      margin-top: 3px;
      font-size: 1.1rem;
    }
    .status {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: .82rem;
      max-height: 260px;
      overflow: auto;
    }
    .error { color: var(--danger); }
    .success { color: var(--accent-dark); }
    .links {
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }
    a { color: var(--accent-dark); font-weight: 650; }
    .empty {
      padding: 30px 14px;
      text-align: center;
      color: var(--muted);
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      aside { position: static; }
      .controls { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 620px) {
      header.top, .toolbar { display: grid; }
      .controls { grid-template-columns: 1fr; }
      .project-head { grid-template-columns: auto auto minmax(0, 1fr); }
      .session-row { grid-template-columns: auto minmax(0, 1fr); }
      .badges { grid-column: 1 / -1; justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <main>
    <header class="top">
      <div>
        <h1>Session Donation Packager</h1>
        <p>Select local Codex and Claude Code project sessions, review estimated volume, then package privacy-filtered output.</p>
      </div>
      <button id="refresh">Refresh scan</button>
    </header>

    <section class="panel">
      <div class="controls">
        <label>Sources
          <input id="sources" type="text" value="codex,claude">
        </label>
        <label>Privacy filter command
          <input id="filterCommand" type="text" value="opf --device cpu --output-mode typed --format json --json-indent 0 --no-print-color-coded-text">
        </label>
        <label class="checkline">
          <input id="includeTools" type="checkbox"> Include tool payloads
        </label>
      </div>
    </section>

    <div class="grid">
      <section>
        <div class="toolbar">
          <div class="left">
            <input id="search" type="search" placeholder="Filter codebases or session IDs">
          </div>
          <div class="right">
            <button id="selectVisible">Select visible</button>
            <button id="clearSelection">Clear</button>
          </div>
        </div>
        <div id="projectList" class="project-list">
          <div class="empty">Scanning local sessions...</div>
        </div>
      </section>

      <aside>
        <section class="panel">
          <strong>Selected donation</strong>
          <div class="metric-grid">
            <div class="metric"><span>Est. tokens</span><strong id="selectedTokens">0</strong></div>
            <div class="metric"><span>Sessions</span><strong id="selectedSessions">0</strong></div>
            <div class="metric"><span>Messages</span><strong id="selectedMessages">0</strong></div>
            <div class="metric"><span>Bytes</span><strong id="selectedBytes">0 B</strong></div>
            <div class="metric"><span>Characters</span><strong id="selectedChars">0</strong></div>
          </div>
        </section>

        <section class="panel">
          <strong>Available data</strong>
          <div class="metric-grid">
            <div class="metric"><span>Codebases</span><strong id="totalProjects">0</strong></div>
            <div class="metric"><span>Sessions</span><strong id="totalSessions">0</strong></div>
            <div class="metric"><span>Est. tokens</span><strong id="totalTokens">0</strong></div>
            <div class="metric"><span>Excluded</span><strong id="excludedCount">0</strong></div>
          </div>
        </section>

        <section class="panel">
          <button id="package" class="primary">Package selected data</button>
          <button id="verify" disabled>Verify last package</button>
          <div id="packageLinks" class="links"></div>
        </section>

        <section id="status" class="status">Ready.</section>
      </aside>
    </div>
  </main>

  <script>
    const state = {
      projects: [],
      sessions: new Map(),
      selected: new Set(),
      expanded: new Set(),
      lastOutputDir: null,
    };

    const el = (id) => document.getElementById(id);

    function formatLarge(n) {
      const value = Number(n || 0);
      const units = [
        ["T", 1e12],
        ["B", 1e9],
        ["M", 1e6],
        ["K", 1e3],
      ];
      for (const [suffix, scale] of units) {
        if (Math.abs(value) >= scale) return `${(value / scale).toFixed(value >= 10 * scale ? 1 : 2)}${suffix}`;
      }
      return Math.round(value).toLocaleString();
    }

    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      const units = ["B", "KB", "MB", "GB", "TB"];
      let n = value;
      let idx = 0;
      while (n >= 1024 && idx < units.length - 1) {
        n /= 1024;
        idx += 1;
      }
      return `${n.toFixed(idx ? 1 : 0)} ${units[idx]}`;
    }

    function setStatus(text, kind = "") {
      el("status").textContent = text;
      el("status").className = `status ${kind}`;
    }

    async function scan(refresh = false) {
      setStatus("Scanning local Codex and Claude sessions...");
      const sources = encodeURIComponent(el("sources").value);
      const includeTools = el("includeTools").checked ? "1" : "0";
      const res = await fetch(`/api/scan?sources=${sources}&include_tools=${includeTools}&refresh=${refresh ? "1" : "0"}`);
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Scan failed");
      state.projects = data.projects || [];
      state.sessions.clear();
      state.selected.clear();
      state.expanded.clear();
      for (const project of state.projects) {
        for (const session of project.sessions) {
          state.sessions.set(session.id, { ...session, projectId: project.id });
        }
      }
      renderProjects();
      updateTotals(data);
      updateSelection();
      setStatus(`Scan complete in ${data.scan_seconds}s. Found ${data.projects.length} codebases and ${data.totals.sessions} eligible sessions.`);
    }

    function updateTotals(data) {
      const excluded = Object.values(data.excluded || {}).reduce((sum, value) => sum + value, 0);
      el("totalProjects").textContent = (data.projects || []).length.toLocaleString();
      el("totalSessions").textContent = Number(data.totals.sessions || 0).toLocaleString();
      el("totalTokens").textContent = formatLarge(data.totals.estimated_tokens || 0);
      el("excludedCount").textContent = excluded.toLocaleString();
    }

    function filteredProjects() {
      const q = el("search").value.trim().toLowerCase();
      return state.projects.filter((project) => {
        if (!q) return true;
        const haystack = `${project.name} ${project.path} ${project.sessions.map((s) => s.id).join(" ")}`.toLowerCase();
        return haystack.includes(q);
      });
    }

    function renderProjects() {
      const list = el("projectList");
      const filtered = filteredProjects();
      if (!filtered.length) {
        list.innerHTML = '<div class="empty">No matching codebases.</div>';
        return;
      }
      list.innerHTML = filtered.map((project) => projectHtml(project)).join("");
      list.querySelectorAll(".project-check").forEach((input) => {
        input.addEventListener("change", () => {
          const project = state.projects.find((item) => item.id === input.dataset.project);
          for (const session of project.sessions) {
            if (input.checked) state.selected.add(session.id);
            else state.selected.delete(session.id);
          }
          renderProjects();
          updateSelection();
        });
      });
      list.querySelectorAll(".disclosure").forEach((button) => {
        button.addEventListener("click", () => {
          const projectId = button.dataset.project;
          if (state.expanded.has(projectId)) state.expanded.delete(projectId);
          else state.expanded.add(projectId);
          renderProjects();
        });
      });
      list.querySelectorAll(".session-check").forEach((input) => {
        input.addEventListener("change", () => {
          if (input.checked) state.selected.add(input.dataset.session);
          else state.selected.delete(input.dataset.session);
          updateProjectChecks();
          updateSelection();
        });
      });
      updateProjectChecks();
    }

    function projectHtml(project) {
      const allSelected = project.sessions.every((session) => state.selected.has(session.id));
      const expanded = state.expanded.has(project.id);
      const stats = project.stats;
      return `
        <article class="project" data-project="${project.id}">
          <div class="project-head">
            <input class="project-check" type="checkbox" data-project="${project.id}" ${allSelected ? "checked" : ""}>
            <button class="disclosure" type="button" data-project="${project.id}" aria-label="${expanded ? "Collapse" : "Expand"} ${escapeHtml(project.name)}">${expanded ? "-" : "+"}</button>
            <div class="project-title">
              <strong>${escapeHtml(project.name)}</strong>
              <small>${escapeHtml(project.path)}</small>
            </div>
            <div class="badges">
              <span class="badge">${formatLarge(stats.estimated_tokens)} est. tokens</span>
              <span class="badge">${formatBytes(stats.bytes)}</span>
              <span class="badge">${stats.sessions.toLocaleString()} sessions</span>
              <span class="badge">${stats.messages.toLocaleString()} messages</span>
            </div>
          </div>
          ${expanded ? `
            <div class="session-list">
              ${project.sessions.slice(0, 20).map((session) => sessionHtml(session)).join("")}
              ${project.sessions.length > 20 ? `<div class="session-main">${project.sessions.length - 20} more sessions included when selected.</div>` : ""}
            </div>
          ` : ""}
        </article>
      `;
    }

    function sessionHtml(session) {
      const stats = session.stats;
      return `
        <div class="session-row">
          <input class="session-check" type="checkbox" data-session="${session.id}" ${state.selected.has(session.id) ? "checked" : ""}>
          <div class="session-main">${escapeHtml(session.source)} - ${escapeHtml(session.id)} - updated ${escapeHtml(session.updated_at || "unknown")}</div>
          <div class="badges">
            <span class="badge">${formatLarge(stats.estimated_tokens)}</span>
            <span class="badge">${stats.messages} msg</span>
          </div>
        </div>
      `;
    }

    function updateProjectChecks() {
      document.querySelectorAll(".project-check").forEach((input) => {
        const project = state.projects.find((item) => item.id === input.dataset.project);
        const selectedCount = project.sessions.filter((session) => state.selected.has(session.id)).length;
        input.checked = selectedCount === project.sessions.length;
        input.indeterminate = selectedCount > 0 && selectedCount < project.sessions.length;
      });
    }

    function updateSelection() {
      const totals = { estimated_tokens: 0, bytes: 0, characters: 0, messages: 0, sessions: 0 };
      for (const sessionId of state.selected) {
        const session = state.sessions.get(sessionId);
        if (!session) continue;
        totals.sessions += 1;
        totals.estimated_tokens += session.stats.estimated_tokens || 0;
        totals.bytes += session.stats.bytes || 0;
        totals.characters += session.stats.characters || 0;
        totals.messages += session.stats.messages || 0;
      }
      el("selectedTokens").textContent = formatLarge(totals.estimated_tokens);
      el("selectedSessions").textContent = totals.sessions.toLocaleString();
      el("selectedMessages").textContent = totals.messages.toLocaleString();
      el("selectedBytes").textContent = formatBytes(totals.bytes);
      el("selectedChars").textContent = totals.characters.toLocaleString();
      el("package").disabled = totals.sessions === 0;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[char]));
    }

    async function packageSelected() {
      setStatus("Packaging selected sessions through the privacy filter...");
      el("package").disabled = true;
      const res = await fetch("/api/package", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sources: el("sources").value,
          include_tools: el("includeTools").checked,
          privacy_filter_command: el("filterCommand").value,
          session_ids: Array.from(state.selected),
        }),
      });
      const data = await res.json();
      el("package").disabled = false;
      if (!data.ok) {
        setStatus(data.error || "Packaging failed", "error");
        return;
      }
      state.lastOutputDir = data.output_dir;
      el("verify").disabled = false;
      el("packageLinks").innerHTML = `
        <a href="${data.review_url}" target="_blank" rel="noreferrer">Open review page</a>
        <a href="${data.download_url}">Download zip package</a>
        <span>${escapeHtml(data.output_dir)}</span>
      `;
      setStatus(`Package ready.\nOutput: ${data.output_dir}\nZip: ${data.zip_path}\nDonation SHA-256: ${data.manifest.files.donation_sha256}`, "success");
    }

    async function verifyLast() {
      if (!state.lastOutputDir) return;
      setStatus("Verifying last package...");
      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_dir: state.lastOutputDir }),
      });
      const data = await res.json();
      if (!data.ok) {
        setStatus(data.output || data.error || "Verify failed", "error");
        return;
      }
      setStatus(data.output, "success");
    }

    el("refresh").addEventListener("click", () => scan(true).catch((err) => setStatus(err.message, "error")));
    el("includeTools").addEventListener("change", () => scan(true).catch((err) => setStatus(err.message, "error")));
    el("sources").addEventListener("change", () => scan(true).catch((err) => setStatus(err.message, "error")));
    el("search").addEventListener("input", renderProjects);
    el("selectVisible").addEventListener("click", () => {
      for (const project of filteredProjects()) {
        for (const session of project.sessions) {
          state.selected.add(session.id);
        }
      }
      renderProjects();
      updateSelection();
    });
    el("clearSelection").addEventListener("click", () => {
      state.selected.clear();
      renderProjects();
      updateSelection();
    });
    el("package").addEventListener("click", packageSelected);
    el("verify").addEventListener("click", verifyLast);

    scan(false).catch((err) => setStatus(err.message, "error"));
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local session donation web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind host, defaults to 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port, defaults to 8765")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DonationWebHandler)
    print(f"Session donation web UI running at http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
