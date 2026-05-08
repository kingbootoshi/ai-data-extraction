#!/usr/bin/env python3
"""
Project-scoped Codex and Claude Code session donation exporter.

This script is intentionally conservative:
- it only donates sessions whose recorded cwd values stay inside one project;
- it omits system prompts and local source file paths from shareable outputs;
- it refuses to write donation data unless an OpenAI Privacy Filter compatible
  command has processed every exported text field.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
PRIVACY_FILTER_MODEL = "openai/privacy-filter"
DEFAULT_OUTPUT_ROOT = "donation_exports"
SHAREABLE_DONATION_FILE = "donation.jsonl"
SHAREABLE_MANIFEST_FILE = "manifest.json"
REVIEW_FILE = "review.html"


SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*API[_-]?KEY[A-Za-z0-9_]*\s*[:=]\s*['\"]?[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]*TOKEN[A-Za-z0-9_]*\s*[:=]\s*['\"]?[^'\"\s,}]+", re.IGNORECASE),
]


@dataclass
class TextFilterResult:
    redacted_text: str
    spans: list[dict[str, Any]]
    span_count: int
    runner: str


@dataclass
class TextSanitizationResult:
    text: str
    privacy_spans: list[dict[str, Any]]
    privacy_span_count: int
    local_replacements: dict[str, int]
    runner: str


@dataclass
class SessionMessage:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class ToolEvent:
    kind: str
    name: str | None
    content: str
    timestamp: str | None = None


@dataclass
class ParsedSession:
    source: str
    source_path: Path
    session_id: str
    cwd_values: list[str]
    messages: list[SessionMessage]
    tool_events: list[ToolEvent]
    created_at: str | None
    updated_at: str | None


@dataclass
class SessionCandidate:
    parsed: ParsedSession | None
    source: str
    source_path: Path
    eligible: bool
    reason: str
    cwd_values: list[str] = field(default_factory=list)


class PrivacyFilterError(RuntimeError):
    """Raised when the required privacy filter cannot process text."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def is_path_inside_project(raw_path: str, project_root: Path) -> bool:
    try:
        candidate = canonical_path(Path(raw_path))
    except (OSError, ValueError):
        return False
    return candidate == project_root or is_relative_to(candidate, project_root)


def collect_cwd_values(value: Any) -> list[str]:
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "cwd" and isinstance(child, str) and child:
                    values.append(child)
                else:
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return values


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def content_to_text(content: Any) -> str:
    parts: list[str] = []

    def append(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value:
                parts.append(value)
            return
        if isinstance(value, list):
            for item in value:
                append(item)
            return
        if isinstance(value, dict):
            item_type = value.get("type")
            if item_type in {"text", "input_text", "output_text"} and isinstance(value.get("text"), str):
                append(value.get("text"))
                return
            if isinstance(value.get("content"), (str, list, dict)):
                append(value.get("content"))
                return
            if isinstance(value.get("message"), str):
                append(value.get("message"))
                return

    append(content)
    return "\n".join(part.strip() for part in parts if part and part.strip()).strip()


def compact_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_codex_session(path: Path, include_tools: bool = False) -> ParsedSession | None:
    messages: list[SessionMessage] = []
    tool_events: list[ToolEvent] = []
    cwd_values: list[str] = []
    session_id = path.stem
    created_at: str | None = None
    updated_at: str | None = None

    for obj in read_jsonl(path):
        timestamp = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None
        if timestamp:
            created_at = created_at or timestamp
            updated_at = timestamp

        cwd_values.extend(collect_cwd_values(obj))
        event_type = obj.get("type")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}

        if event_type == "session_meta":
            if isinstance(payload.get("id"), str):
                session_id = payload["id"]
            if isinstance(payload.get("timestamp"), str):
                created_at = created_at or payload["timestamp"]
            continue

        if event_type == "response_item":
            payload_type = payload.get("type")
            if payload_type == "message":
                role = payload.get("role")
                text = content_to_text(payload.get("content"))
                if role in {"user", "assistant"} and text:
                    messages.append(SessionMessage(role=role, content=text, timestamp=timestamp))
            elif include_tools and payload_type in {
                "function_call",
                "function_call_output",
                "local_shell_call",
                "web_search_call",
            }:
                tool_events.append(
                    ToolEvent(
                        kind=str(payload_type),
                        name=payload.get("name") if isinstance(payload.get("name"), str) else None,
                        content=compact_json_text(payload),
                        timestamp=timestamp,
                    )
                )
            continue

        if event_type == "event_msg":
            payload_type = payload.get("type")
            if payload_type in {"user_message", "agent_message"}:
                text = payload.get("message") if isinstance(payload.get("message"), str) else ""
                role = "assistant" if payload_type == "agent_message" else "user"
                if text.strip():
                    messages.append(SessionMessage(role=role, content=text.strip(), timestamp=timestamp))
            elif include_tools and payload_type in {"tool_use", "tool_result", "diff", "exec_command_end"}:
                tool_events.append(
                    ToolEvent(
                        kind=str(payload_type),
                        name=payload.get("tool") if isinstance(payload.get("tool"), str) else None,
                        content=compact_json_text(payload),
                        timestamp=timestamp,
                    )
                )

    if not messages and not tool_events:
        return None

    return ParsedSession(
        source="codex",
        source_path=path,
        session_id=session_id,
        cwd_values=dedupe_keep_order(cwd_values),
        messages=messages,
        tool_events=tool_events,
        created_at=created_at,
        updated_at=updated_at,
    )


def parse_claude_session(path: Path, include_tools: bool = False) -> ParsedSession | None:
    messages: list[SessionMessage] = []
    tool_events: list[ToolEvent] = []
    cwd_values: list[str] = []
    session_id = path.stem
    created_at: str | None = None
    updated_at: str | None = None

    for obj in read_jsonl(path):
        timestamp = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None
        if timestamp:
            created_at = created_at or timestamp
            updated_at = timestamp

        if isinstance(obj.get("sessionId"), str):
            session_id = obj["sessionId"]

        cwd_values.extend(collect_cwd_values(obj))
        event_type = obj.get("type")

        if event_type in {"user", "assistant"}:
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            role = message.get("role") if message.get("role") in {"user", "assistant"} else event_type
            text = content_to_text(message.get("content"))
            if text:
                messages.append(SessionMessage(role=role, content=text, timestamp=timestamp))

            if include_tools:
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in {"tool_use", "tool_result"}:
                            tool_events.append(
                                ToolEvent(
                                    kind=str(item.get("type")),
                                    name=item.get("name") if isinstance(item.get("name"), str) else None,
                                    content=compact_json_text(item),
                                    timestamp=timestamp,
                                )
                            )
            continue

        if include_tools and event_type == "tool_result":
            tool_events.append(
                ToolEvent(
                    kind="tool_result",
                    name=None,
                    content=compact_json_text(obj.get("toolResult", obj)),
                    timestamp=timestamp,
                )
            )

    if not messages and not tool_events:
        return None

    return ParsedSession(
        source="claude-code",
        source_path=path,
        session_id=session_id,
        cwd_values=dedupe_keep_order(cwd_values),
        messages=messages,
        tool_events=tool_events,
        created_at=created_at,
        updated_at=updated_at,
    )


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def session_eligibility(parsed: ParsedSession | None, project_root: Path) -> tuple[bool, str, list[str]]:
    if parsed is None:
        return False, "no exportable user or assistant messages", []

    cwd_values = parsed.cwd_values
    if not cwd_values:
        return False, "no cwd metadata found", []

    outside = [cwd for cwd in cwd_values if not is_path_inside_project(cwd, project_root)]
    if outside:
        return False, "one or more cwd values are outside the selected project", cwd_values

    return True, "eligible", cwd_values


def codex_installations() -> list[Path]:
    home = Path.home()
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        env_path = canonical_path(Path(env_home))
        return [env_path] if env_path.exists() else []

    candidates = [
        home / ".codex",
        home / ".codex-local",
    ]
    return dedupe_paths(path for path in candidates if path and path.exists())


def claude_installations() -> list[Path]:
    home = Path.home()
    env_home = os.environ.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_HOME")
    if env_home:
        env_path = canonical_path(Path(env_home))
        return [env_path] if env_path.exists() else []

    candidates = [
        home / ".claude",
        home / ".claude-code",
        home / ".claude-local",
        home / ".claude-m2",
        home / ".claude-zai",
    ]
    return dedupe_paths(path for path in candidates if path and path.exists())


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(canonical_path(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical_path(path))
    return result


def discover_codex_session_files(installation: Path) -> list[Path]:
    sessions_dir = installation / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_claude_session_files(installation: Path) -> list[Path]:
    projects_dir = installation / "projects"
    if not projects_dir.exists():
        return []
    return sorted(
        (path for path in projects_dir.rglob("*.jsonl") if not path.name.startswith("agent-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def discover_candidates(project_root: Path, sources: set[str], include_tools: bool) -> list[SessionCandidate]:
    candidates: list[SessionCandidate] = []

    if "codex" in sources:
        for installation in codex_installations():
            for path in discover_codex_session_files(installation):
                parsed = parse_codex_session(path, include_tools=include_tools)
                eligible, reason, cwds = session_eligibility(parsed, project_root)
                candidates.append(
                    SessionCandidate(
                        parsed=parsed,
                        source="codex",
                        source_path=path,
                        eligible=eligible,
                        reason=reason,
                        cwd_values=cwds,
                    )
                )

    if "claude" in sources or "claude-code" in sources:
        for installation in claude_installations():
            for path in discover_claude_session_files(installation):
                parsed = parse_claude_session(path, include_tools=include_tools)
                eligible, reason, cwds = session_eligibility(parsed, project_root)
                candidates.append(
                    SessionCandidate(
                        parsed=parsed,
                        source="claude-code",
                        source_path=path,
                        eligible=eligible,
                        reason=reason,
                        cwd_values=cwds,
                    )
                )

    return candidates


class LocalMinimizer:
    def __init__(self, project_root: Path):
        self.project_root = canonical_path(project_root)
        self.home = canonical_path(Path.home())
        self.project_aliases = path_aliases(self.project_root)
        self.home_aliases = path_aliases(self.home)

    def apply(self, text: str) -> tuple[str, dict[str, int]]:
        counts = {
            "project_paths": 0,
            "home_paths": 0,
            "absolute_paths": 0,
            "secret_patterns": 0,
        }
        minimized = text

        for project_path in self.project_aliases:
            project_pattern = re.escape(str(project_path))
            minimized, count = re.subn(project_pattern + r"(?=/|[\s'\"),:\]}]|$)", "<PROJECT>", minimized)
            counts["project_paths"] += count

        for home_path in self.home_aliases:
            home_pattern = re.escape(str(home_path))
            minimized, count = re.subn(home_pattern + r"(?=/|[\s'\"),:\]}]|$)", "<HOME>", minimized)
            counts["home_paths"] += count

        absolute_path_pattern = re.compile(r"(?<![\w<])/(?:Users|private|tmp|var|home)/[^\s'\"),:\]}]+")
        minimized, count = absolute_path_pattern.subn("<LOCAL_PATH>", minimized)
        counts["absolute_paths"] += count

        for pattern in SECRET_PATTERNS:
            minimized, count = pattern.subn("<SECRET>", minimized)
            counts["secret_patterns"] += count

        return minimized, counts


def path_aliases(path: Path) -> list[str]:
    value = str(path)
    aliases = [value]
    if value.startswith("/private/var/"):
        aliases.append(value.removeprefix("/private"))
    elif value.startswith("/var/"):
        aliases.append(f"/private{value}")
    return dedupe_keep_order(aliases)


class PrivacyFilter:
    def __init__(self, command: str | None = None):
        self.command = command or "opf --output-mode typed"
        self.command_parts = shlex.split(self.command)
        if not self.command_parts:
            raise PrivacyFilterError("privacy filter command is empty")

    @property
    def runner(self) -> str:
        return Path(self.command_parts[0]).name

    def ensure_available(self) -> None:
        executable = self.command_parts[0]
        if shutil.which(executable) is None and not Path(executable).exists():
            raise PrivacyFilterError(
                "OpenAI Privacy Filter command not found. Install https://github.com/openai/privacy-filter "
                "so `opf` is available, or pass --privacy-filter-command to a compatible local wrapper."
            )

    def filter_text(self, text: str) -> TextFilterResult:
        self.ensure_available()
        if not text:
            return TextFilterResult(redacted_text=text, spans=[], span_count=0, runner=self.runner)

        proc = subprocess.run(
            self.command_parts,
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise PrivacyFilterError(
                f"privacy filter command failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )

        payloads = parse_filter_payloads(proc.stdout)
        if not payloads:
            raise PrivacyFilterError("privacy filter produced no parseable JSON output")

        redacted_parts: list[str] = []
        spans: list[dict[str, Any]] = []
        for payload in payloads:
            redacted = payload.get("redacted_text")
            if not isinstance(redacted, str):
                raise PrivacyFilterError("privacy filter JSON output did not include redacted_text")
            redacted_parts.append(redacted)
            detected_spans = payload.get("detected_spans", [])
            if isinstance(detected_spans, list):
                spans.extend(safe_span(span) for span in detected_spans if isinstance(span, dict))

        return TextFilterResult(
            redacted_text="\n".join(redacted_parts),
            spans=spans,
            span_count=len(spans),
            runner=self.runner,
        )


def parse_filter_payloads(stdout: str) -> list[dict[str, Any]]:
    stripped = stdout.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    payloads: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def safe_span(span: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in ("label", "entity_group", "start", "end", "placeholder", "score"):
        if key in span:
            safe[key] = span[key]
    if "label" not in safe and "entity_group" in safe:
        safe["label"] = safe["entity_group"]
    return safe


def sanitize_text(text: str, minimizer: LocalMinimizer, privacy_filter: PrivacyFilter) -> TextSanitizationResult:
    minimized, replacements = minimizer.apply(text)
    filtered = privacy_filter.filter_text(minimized)
    return TextSanitizationResult(
        text=filtered.redacted_text,
        privacy_spans=filtered.spans,
        privacy_span_count=filtered.span_count,
        local_replacements=replacements,
        runner=filtered.runner,
    )


def candidate_hash(candidate: SessionCandidate) -> str:
    parsed_id = candidate.parsed.session_id if candidate.parsed else candidate.source_path.stem
    return stable_hash(f"{candidate.source}\n{canonical_path(candidate.source_path)}\n{parsed_id}")[:16]


def export_session(
    candidate: SessionCandidate,
    project_hash: str,
    minimizer: LocalMinimizer,
    privacy_filter: PrivacyFilter,
    include_tools: bool,
) -> dict[str, Any]:
    if candidate.parsed is None:
        raise ValueError("cannot export an unparsed session")

    parsed = candidate.parsed
    session_hash = candidate_hash(candidate)
    messages: list[dict[str, Any]] = []
    total_spans = 0
    total_replacements: dict[str, int] = {
        "project_paths": 0,
        "home_paths": 0,
        "absolute_paths": 0,
        "secret_patterns": 0,
    }

    for message in parsed.messages:
        sanitized = sanitize_text(message.content, minimizer, privacy_filter)
        total_spans += sanitized.privacy_span_count
        merge_counts(total_replacements, sanitized.local_replacements)
        messages.append(
            {
                "role": message.role,
                "content": sanitized.text,
                "timestamp": message.timestamp,
                "privacy_spans": sanitized.privacy_spans,
                "local_minimization": sanitized.local_replacements,
                "privacy_filter": {
                    "model": PRIVACY_FILTER_MODEL,
                    "runner": sanitized.runner,
                    "status": "filtered",
                },
            }
        )

    tool_events: list[dict[str, Any]] = []
    if include_tools:
        for event in parsed.tool_events:
            sanitized = sanitize_text(event.content, minimizer, privacy_filter)
            total_spans += sanitized.privacy_span_count
            merge_counts(total_replacements, sanitized.local_replacements)
            tool_events.append(
                {
                    "kind": event.kind,
                    "name": event.name,
                    "content": sanitized.text,
                    "timestamp": event.timestamp,
                    "privacy_spans": sanitized.privacy_spans,
                    "local_minimization": sanitized.local_replacements,
                    "privacy_filter": {
                        "model": PRIVACY_FILTER_MODEL,
                        "runner": sanitized.runner,
                        "status": "filtered",
                    },
                }
            )

    record = {
        "schema_version": SCHEMA_VERSION,
        "source": parsed.source,
        "session_hash": session_hash,
        "project_hash": project_hash,
        "created_at": parsed.created_at,
        "updated_at": parsed.updated_at,
        "message_count": len(messages),
        "tool_event_count": len(tool_events),
        "messages": messages,
        "privacy_filter": {
            "model": PRIVACY_FILTER_MODEL,
            "runner": privacy_filter.runner,
            "status": "filtered",
            "filtered_text_fields": len(messages) + len(tool_events),
            "span_count": total_spans,
        },
        "local_minimization": total_replacements,
    }
    if tool_events:
        record["tool_events"] = tool_events
    return record


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def export_donation(
    project_root: Path,
    candidates: list[SessionCandidate],
    output_dir: Path,
    privacy_filter: PrivacyFilter,
    include_tools: bool,
    selected_hashes: set[str] | None = None,
) -> Path:
    eligible = [candidate for candidate in candidates if candidate.eligible and candidate.parsed]
    if selected_hashes is not None:
        eligible = [candidate for candidate in eligible if candidate_hash(candidate) in selected_hashes]

    if not eligible:
        raise SystemExit("No eligible sessions selected for donation.")

    privacy_filter.ensure_available()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_hash = stable_hash(str(project_root))
    minimizer = LocalMinimizer(project_root)
    records = [
        export_session(candidate, project_hash, minimizer, privacy_filter, include_tools)
        for candidate in eligible
    ]

    donation_path = output_dir / SHAREABLE_DONATION_FILE
    with donation_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = build_manifest(project_root, candidates, eligible, output_dir, donation_path, privacy_filter, include_tools)
    manifest_path = output_dir / SHAREABLE_MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    review_path = output_dir / REVIEW_FILE
    review_path.write_text(render_review_html(manifest, records), encoding="utf-8")

    return output_dir


def build_manifest(
    project_root: Path,
    candidates: list[SessionCandidate],
    donated: list[SessionCandidate],
    output_dir: Path,
    donation_path: Path,
    privacy_filter: PrivacyFilter,
    include_tools: bool,
) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate.eligible and candidate.parsed]
    excluded = [candidate for candidate in candidates if not candidate.eligible]
    donation_hash = file_sha256(donation_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "project": {
            "root_hash": stable_hash(str(project_root)),
            "name_hint": project_root.name,
        },
        "privacy_filter": {
            "model": PRIVACY_FILTER_MODEL,
            "runner": privacy_filter.runner,
            "status": "filtered",
        },
        "options": {
            "include_tools": include_tools,
        },
        "counts": {
            "candidates": len(candidates),
            "eligible": len(eligible),
            "donated": len(donated),
            "excluded": len(excluded),
        },
        "files": {
            "donation_jsonl": SHAREABLE_DONATION_FILE,
            "donation_sha256": donation_hash,
            "review_html": REVIEW_FILE,
        },
        "donated_sessions": [
            {
                "source": candidate.source,
                "session_hash": candidate_hash(candidate),
                "message_count": len(candidate.parsed.messages) if candidate.parsed else 0,
                "tool_event_count": len(candidate.parsed.tool_events) if candidate.parsed else 0,
                "created_at": candidate.parsed.created_at if candidate.parsed else None,
                "updated_at": candidate.parsed.updated_at if candidate.parsed else None,
            }
            for candidate in donated
        ],
        "excluded_summary": summarize_exclusions(excluded),
        "verify_command": "python3 donate_project_sessions.py verify <output-directory>",
    }


def summarize_exclusions(excluded: list[SessionCandidate]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for candidate in excluded:
        summary[candidate.reason] = summary.get(candidate.reason, 0) + 1
    return summary


def render_review_html(manifest: dict[str, Any], records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        messages_html = []
        for message in record.get("messages", []):
            content = html.escape(message.get("content", ""))
            role = html.escape(message.get("role", "message"))
            spans = len(message.get("privacy_spans", []))
            replacements = sum(int(v) for v in message.get("local_minimization", {}).values())
            messages_html.append(
                f"""
                <article class="message" data-role="{role}">
                  <header><span>{role}</span><small>{spans} privacy spans, {replacements} local replacements</small></header>
                  <pre>{content}</pre>
                </article>
                """
            )
        rows.append(
            f"""
            <section class="session" data-source="{html.escape(record.get("source", ""))}">
              <h2>{html.escape(record.get("source", ""))} - {html.escape(record.get("session_hash", ""))}</h2>
              <dl>
                <div><dt>Messages</dt><dd>{record.get("message_count", 0)}</dd></div>
                <div><dt>Privacy spans</dt><dd>{record.get("privacy_filter", {}).get("span_count", 0)}</dd></div>
                <div><dt>Local replacements</dt><dd>{sum(int(v) for v in record.get("local_minimization", {}).values())}</dd></div>
              </dl>
              {''.join(messages_html)}
            </section>
            """
        )

    counts = manifest.get("counts", {})
    donation_hash = manifest.get("files", {}).get("donation_sha256", "")
    created_at = manifest.get("created_at", "")
    project_hint = manifest.get("project", {}).get("name_hint", "project")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Session Donation Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --ink: #17202a;
      --muted: #617083;
      --line: #d8dee8;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #a16207;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    header.page {{
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.6rem, 2.4vw, 2.4rem);
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric, .session {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{
      padding: 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: .82rem;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.25rem;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin: 20px 0 14px;
    }}
    input {{
      width: min(460px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }}
    .session {{
      padding: 16px;
      margin: 14px 0;
    }}
    .session h2 {{
      margin: 0 0 12px;
      font-size: 1rem;
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin: 0 0 14px;
    }}
    dl div {{
      border-left: 3px solid var(--accent);
      padding-left: 8px;
    }}
    dt {{
      color: var(--muted);
      font-size: .8rem;
    }}
    dd {{
      margin: 2px 0 0;
      font-weight: 650;
    }}
    .message {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }}
    .message header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .message span {{
      color: var(--accent);
      font-weight: 700;
      text-transform: capitalize;
    }}
    .message small, .note {{
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: .88rem;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <main>
    <header class="page">
      <h1>Session Donation Review</h1>
      <p class="note">Project hint: {html.escape(project_hint)}. Created {html.escape(created_at)}. Donation SHA-256: <code>{html.escape(donation_hash)}</code>.</p>
    </header>
    <section class="summary" aria-label="Summary">
      <div class="metric"><span>Candidates</span><strong>{counts.get("candidates", 0)}</strong></div>
      <div class="metric"><span>Eligible</span><strong>{counts.get("eligible", 0)}</strong></div>
      <div class="metric"><span>Donated</span><strong>{counts.get("donated", 0)}</strong></div>
      <div class="metric"><span>Excluded</span><strong>{counts.get("excluded", 0)}</strong></div>
    </section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Filter review text or session hash" autocomplete="off">
    </div>
    <div id="sessions">
      {''.join(rows)}
    </div>
  </main>
  <script>
    const search = document.querySelector("#search");
    const sessions = Array.from(document.querySelectorAll(".session"));
    search.addEventListener("input", () => {{
      const q = search.value.trim().toLowerCase();
      for (const session of sessions) {{
        session.classList.toggle("hidden", q && !session.textContent.toLowerCase().includes(q));
      }}
    }});
  </script>
</body>
</html>
"""


def parse_sources(raw_sources: str) -> set[str]:
    sources = {source.strip().lower() for source in raw_sources.split(",") if source.strip()}
    normalized: set[str] = set()
    for source in sources:
        if source in {"claude", "claude-code"}:
            normalized.add("claude")
        elif source == "codex":
            normalized.add("codex")
        else:
            raise SystemExit(f"Unknown source: {source}. Use codex, claude, or both.")
    if not normalized:
        raise SystemExit("At least one source is required.")
    return normalized


def print_candidate_list(candidates: list[SessionCandidate]) -> None:
    eligible = [candidate for candidate in candidates if candidate.eligible and candidate.parsed]
    excluded = [candidate for candidate in candidates if not candidate.eligible]

    print(f"Eligible sessions: {len(eligible)}")
    for index, candidate in enumerate(eligible, 1):
        parsed = candidate.parsed
        assert parsed is not None
        print(
            f"  {index:>3}. {candidate_hash(candidate)}  "
            f"{parsed.source:11}  messages={len(parsed.messages):>4}  "
            f"tools={len(parsed.tool_events):>4}  updated={parsed.updated_at or 'unknown'}"
        )

    if excluded:
        print()
        print("Excluded sessions:")
        for reason, count in sorted(summarize_exclusions(excluded).items()):
            print(f"  {count:>4}  {reason}")


def prompt_for_selection(candidates: list[SessionCandidate]) -> set[str]:
    eligible = [candidate for candidate in candidates if candidate.eligible and candidate.parsed]
    if not eligible:
        return set()

    print()
    answer = input("Donate which eligible sessions? Type 'all', a comma-separated number list, or 'none': ").strip()
    if answer.lower() == "all":
        return {candidate_hash(candidate) for candidate in eligible}
    if answer.lower() in {"none", "no", "n", ""}:
        return set()

    selected: set[str] = set()
    by_index = {str(index): candidate for index, candidate in enumerate(eligible, 1)}
    for raw_part in answer.split(","):
        part = raw_part.strip()
        candidate = by_index.get(part)
        if candidate is None:
            raise SystemExit(f"Invalid selection: {part}")
        selected.add(candidate_hash(candidate))
    return selected


def default_output_dir(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_root.name).strip("-") or "project"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{safe_name}_{timestamp}"


def command_export(args: argparse.Namespace) -> int:
    project_root = canonical_path(Path(args.project))
    sources = parse_sources(args.sources)
    candidates = discover_candidates(project_root, sources, include_tools=args.include_tools)

    print(f"Project: {project_root}")
    print_candidate_list(candidates)

    if args.list:
        return 0

    selected_hashes: set[str] | None = None
    if args.session:
        selected_hashes = set(args.session)
    elif not args.yes:
        selected_hashes = prompt_for_selection(candidates)
        if not selected_hashes:
            print("No sessions selected.")
            return 1

    output_dir = canonical_path(Path(args.output_dir)) if args.output_dir else canonical_path(default_output_dir(project_root))
    privacy_filter = PrivacyFilter(args.privacy_filter_command)
    export_donation(
        project_root=project_root,
        candidates=candidates,
        output_dir=output_dir,
        privacy_filter=privacy_filter,
        include_tools=args.include_tools,
        selected_hashes=selected_hashes,
    )

    print()
    print(f"Wrote shareable donation data to: {output_dir / SHAREABLE_DONATION_FILE}")
    print(f"Wrote verification manifest to: {output_dir / SHAREABLE_MANIFEST_FILE}")
    print(f"Wrote local review page to: {output_dir / REVIEW_FILE}")
    print(f"Verify with: python3 donate_project_sessions.py verify {output_dir}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    output_dir = canonical_path(Path(args.output_dir))
    manifest_path = output_dir / SHAREABLE_MANIFEST_FILE
    donation_path = output_dir / SHAREABLE_DONATION_FILE
    review_path = output_dir / REVIEW_FILE

    failures: list[str] = []
    if not manifest_path.exists():
        failures.append("manifest.json is missing")
    if not donation_path.exists():
        failures.append("donation.jsonl is missing")
    if not review_path.exists():
        failures.append("review.html is missing")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("files", {}).get("donation_sha256")
    actual_hash = file_sha256(donation_path)
    if expected_hash != actual_hash:
        failures.append("donation.jsonl hash does not match manifest")

    records: list[dict[str, Any]] = []
    with donation_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"line {line_number} is not valid JSON: {exc}")
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                failures.append(f"line {line_number} is not a JSON object")

    donated_count = manifest.get("counts", {}).get("donated")
    if donated_count != len(records):
        failures.append("donated record count does not match manifest")

    for index, record in enumerate(records, 1):
        filter_meta = record.get("privacy_filter", {})
        if filter_meta.get("model") != PRIVACY_FILTER_MODEL or filter_meta.get("status") != "filtered":
            failures.append(f"record {index} is missing successful OpenAI Privacy Filter metadata")
        if "source_path" in record or "source_file" in record:
            failures.append(f"record {index} contains a raw source path field")
        for message_index, message in enumerate(record.get("messages", []), 1):
            message_filter = message.get("privacy_filter", {})
            if message_filter.get("model") != PRIVACY_FILTER_MODEL or message_filter.get("status") != "filtered":
                failures.append(f"record {index} message {message_index} is missing privacy filter metadata")

    home_path = str(Path.home())
    for checked_path in (donation_path, manifest_path, review_path):
        checked_text = checked_path.read_text(encoding="utf-8")
        if home_path in checked_text:
            failures.append(f"{checked_path.name} contains the user's home directory path")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(f"OK: {len(records)} donation record(s) verified")
    print(f"OK: donation SHA-256 {actual_hash}")
    print(f"OK: review page present at {review_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export project-scoped Codex and Claude Code sessions through OpenAI Privacy Filter."
    )
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export", help="discover, filter, and export project sessions")
    export_parser.add_argument("--project", default=os.getcwd(), help="project root to scope sessions to")
    export_parser.add_argument("--sources", default="codex,claude", help="comma-separated sources: codex,claude")
    export_parser.add_argument("--output-dir", help="output directory for donation.jsonl, manifest.json, and review.html")
    export_parser.add_argument(
        "--privacy-filter-command",
        default=None,
        help="command compatible with `opf --output-mode typed`; defaults to `opf --output-mode typed`",
    )
    export_parser.add_argument("--include-tools", action="store_true", help="include filtered tool call/result payloads")
    export_parser.add_argument("--session", action="append", help="eligible session hash to include; repeatable")
    export_parser.add_argument("--list", action="store_true", help="list eligible/excluded sessions without exporting")
    export_parser.add_argument("--yes", action="store_true", help="export all eligible sessions without an interactive prompt")
    export_parser.set_defaults(func=command_export)

    verify_parser = subparsers.add_parser("verify", help="verify an exported donation directory")
    verify_parser.add_argument("output_dir", help="directory containing donation.jsonl and manifest.json")
    verify_parser.set_defaults(func=command_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args:
        parser.print_help()
        return 0

    command_names = {"export", "verify", "-h", "--help"}
    if raw_args and raw_args[0] not in command_names:
        raw_args = ["export", *raw_args]
    args = parser.parse_args(raw_args)

    try:
        return args.func(args)
    except PrivacyFilterError as exc:
        print(f"Privacy filter error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
