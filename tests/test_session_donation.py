import argparse
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import donate_project_sessions as donation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_fake_privacy_filter(root: Path) -> str:
    script = root / "fake_privacy_filter.py"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import re
            import sys

            text = sys.stdin.read()
            replacements = [
                ("Alice", "<PRIVATE_PERSON>", "private_person"),
                ("alice@example.com", "<PRIVATE_EMAIL>", "private_email"),
            ]
            spans = []
            redacted = text
            for needle, replacement, label in replacements:
                for match in re.finditer(re.escape(needle), text):
                    spans.append({
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "text": needle,
                        "placeholder": replacement,
                    })
                redacted = redacted.replace(needle, replacement)

            print(json.dumps({
                "schema_version": 1,
                "summary": {"span_count": len(spans)},
                "text": text,
                "detected_spans": spans,
                "redacted_text": redacted,
            }))
            """
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return f"{sys.executable} {script}"


class SessionDonationTests(unittest.TestCase):
    def test_opf_cli_uses_text_file_for_multiline_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opf = root / "opf"
            opf.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import sys
                    from pathlib import Path

                    text_file = sys.argv[sys.argv.index("--text-file") + 1]
                    text = Path(text_file).read_text(encoding="utf-8")
                    print(json.dumps({
                        "schema_version": 1,
                        "summary": {"span_count": 0},
                        "text": text,
                        "detected_spans": [],
                        "redacted_text": text.replace("Alice", "<PRIVATE_PERSON>"),
                    }, indent=0))
                    """
                ),
                encoding="utf-8",
            )
            opf.chmod(opf.stat().st_mode | stat.S_IXUSR)

            result = donation.PrivacyFilter(str(opf)).filter_text("hello Alice\nsecond line")

            self.assertEqual(result.redacted_text, "hello <PRIVATE_PERSON>\nsecond line")

    def test_codex_discovery_only_marks_project_scoped_sessions_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            other = root / "other"
            project.mkdir()
            other.mkdir()
            codex_home = root / "codex"

            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-eligible.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "payload": {"id": "eligible", "cwd": str(project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T10:00:01Z",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Help with this project"}],
                        },
                    },
                ],
            )
            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-outside.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T11:00:00Z",
                        "payload": {"id": "outside", "cwd": str(other)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T11:00:01Z",
                        "payload": {"type": "message", "role": "user", "content": "Wrong project"},
                    },
                ],
            )
            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-mixed.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T12:00:00Z",
                        "payload": {"id": "mixed", "cwd": str(project)},
                    },
                    {
                        "type": "turn_context",
                        "timestamp": "2026-05-08T12:00:01Z",
                        "payload": {"cwd": str(other), "model": "gpt"},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T12:00:02Z",
                        "payload": {"type": "message", "role": "user", "content": "Cross project"},
                    },
                ],
            )

            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home), "CLAUDE_CONFIG_DIR": str(root / "empty")}):
                candidates = donation.discover_candidates(
                    donation.canonical_path(project),
                    {"codex"},
                    include_tools=False,
                )

            eligible = [candidate for candidate in candidates if candidate.eligible]
            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible[0].parsed.session_id, "eligible")

            excluded_reasons = sorted(candidate.reason for candidate in candidates if not candidate.eligible)
            self.assertEqual(
                excluded_reasons,
                [
                    "one or more cwd values are outside the selected project",
                    "one or more cwd values are outside the selected project",
                ],
            )

    def test_claude_sessions_are_scoped_by_recorded_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            claude_home = root / "claude"
            write_jsonl(
                claude_home / "projects/-tmp-project/session-1.jsonl",
                [
                    {
                        "type": "user",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                        "message": {"role": "user", "content": "Claude project message"},
                    },
                    {
                        "type": "assistant",
                        "timestamp": "2026-05-08T10:00:01Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Claude project answer"}],
                        },
                    },
                ],
            )

            with patch.dict(os.environ, {"CODEX_HOME": str(root / "empty"), "CLAUDE_CONFIG_DIR": str(claude_home)}):
                candidates = donation.discover_candidates(
                    donation.canonical_path(project),
                    {"claude"},
                    include_tools=False,
                )

            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0].eligible)
            self.assertEqual(candidates[0].parsed.source, "claude-code")
            self.assertEqual(len(candidates[0].parsed.messages), 2)

    def test_export_filters_text_and_verify_accepts_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            codex_home = root / "codex"
            output_dir = root / "out"
            filter_command = make_fake_privacy_filter(root)

            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-private.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "payload": {"id": "private", "cwd": str(project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T10:00:01Z",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": f"Alice can be reached at alice@example.com in {project}/src/app.py",
                        },
                    },
                ],
            )

            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home), "CLAUDE_CONFIG_DIR": str(root / "empty")}):
                candidates = donation.discover_candidates(
                    donation.canonical_path(project),
                    {"codex"},
                    include_tools=False,
                )
                donation.export_donation(
                    project_root=donation.canonical_path(project),
                    candidates=candidates,
                    output_dir=output_dir,
                    privacy_filter=donation.PrivacyFilter(filter_command),
                    include_tools=False,
                )

            donation_text = (output_dir / donation.SHAREABLE_DONATION_FILE).read_text(encoding="utf-8")
            manifest_text = (output_dir / donation.SHAREABLE_MANIFEST_FILE).read_text(encoding="utf-8")
            review_text = (output_dir / donation.REVIEW_FILE).read_text(encoding="utf-8")
            self.assertIn("<PRIVATE_PERSON>", donation_text)
            self.assertIn("<PRIVATE_EMAIL>", donation_text)
            self.assertIn("<PROJECT>", donation_text)
            self.assertNotIn("Alice", donation_text)
            self.assertNotIn("alice@example.com", donation_text)
            self.assertNotIn(str(project), donation_text)
            self.assertNotIn(str(Path.home()), donation_text)
            self.assertNotIn(str(Path.home()), manifest_text)
            self.assertNotIn(str(Path.home()), review_text)

            verify_code = donation.command_verify(argparse.Namespace(output_dir=str(output_dir)))
            self.assertEqual(verify_code, 0)

    def test_verify_rejects_unminimized_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            donation_path = output_dir / donation.SHAREABLE_DONATION_FILE
            manifest_path = output_dir / donation.SHAREABLE_MANIFEST_FILE
            review_path = output_dir / donation.REVIEW_FILE
            record = {
                "schema_version": donation.SCHEMA_VERSION,
                "source": "codex",
                "session_hash": "session",
                "messages": [
                    {
                        "role": "user",
                        "content": "see /Users/example/project/file.py",
                        "privacy_filter": {
                            "model": donation.PRIVACY_FILTER_MODEL,
                            "status": "filtered",
                        },
                    }
                ],
                "privacy_filter": {
                    "model": donation.PRIVACY_FILTER_MODEL,
                    "status": "filtered",
                },
            }
            donation_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = {
                "counts": {"donated": 1},
                "files": {"donation_sha256": donation.file_sha256(donation_path)},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            review_path.write_text("review", encoding="utf-8")

            verify_code = donation.command_verify(argparse.Namespace(output_dir=str(output_dir)))

            self.assertEqual(verify_code, 1)


if __name__ == "__main__":
    unittest.main()
