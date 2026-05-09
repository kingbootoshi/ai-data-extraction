import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import donate_project_sessions as donation
import session_donation_web as web
from test_session_donation import make_fake_privacy_filter, write_jsonl


class SessionDonationWebTests(unittest.TestCase):
    def test_build_project_index_groups_sessions_and_counts_estimated_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            (project / ".git").mkdir(parents=True)
            codex_home = root / "codex"

            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-counts.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "payload": {"id": "counts", "cwd": str(project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T10:00:01Z",
                        "payload": {"type": "message", "role": "user", "content": "abcd" * 10},
                    },
                ],
            )

            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home), "CLAUDE_CONFIG_DIR": str(root / "empty")}):
                index = web.build_project_index({"codex"}, include_tools=False)

            public = web.public_index(index)
            self.assertEqual(len(public["projects"]), 1)
            self.assertEqual(public["totals"]["sessions"], 1)
            self.assertEqual(public["totals"]["characters"], 40)
            self.assertEqual(public["totals"]["estimated_tokens"], 10)
            self.assertEqual(public["projects"][0]["path"], str(donation.canonical_path(project)))

    def test_public_index_sorts_projects_and_sessions_by_estimated_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big_project = root / "z-big"
            small_project = root / "a-small"
            (big_project / ".git").mkdir(parents=True)
            (small_project / ".git").mkdir(parents=True)
            codex_home = root / "codex"

            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-big-small.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T12:00:00Z",
                        "payload": {"id": "big-small", "cwd": str(big_project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T12:00:01Z",
                        "payload": {"type": "message", "role": "user", "content": "x" * 40},
                    },
                ],
            )
            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-big-large.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "payload": {"id": "big-large", "cwd": str(big_project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T10:00:01Z",
                        "payload": {"type": "message", "role": "user", "content": "x" * 400},
                    },
                ],
            )
            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-small-project.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T11:00:00Z",
                        "payload": {"id": "small-project", "cwd": str(small_project)},
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-05-08T11:00:01Z",
                        "payload": {"type": "message", "role": "user", "content": "x" * 80},
                    },
                ],
            )

            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home), "CLAUDE_CONFIG_DIR": str(root / "empty")}):
                public = web.public_index(web.build_project_index({"codex"}, include_tools=False))

            self.assertEqual(public["projects"][0]["path"], str(donation.canonical_path(big_project)))
            self.assertEqual(public["projects"][1]["path"], str(donation.canonical_path(small_project)))
            session_tokens = [session["stats"]["estimated_tokens"] for session in public["projects"][0]["sessions"]]
            self.assertEqual(session_tokens, [100, 10])

    def test_web_ui_shows_selected_donation_without_collector_progress(self) -> None:
        self.assertIn("Selected donation", web.HTML_PAGE)
        self.assertNotIn('id="target"', web.HTML_PAGE)
        self.assertNotIn("Progress", web.HTML_PAGE)
        self.assertNotIn("progressBar", web.HTML_PAGE)

    def test_web_ui_collapses_projects_by_default_for_project_level_selection(self) -> None:
        self.assertIn("expanded: new Set()", web.HTML_PAGE)
        self.assertIn("state.expanded.clear()", web.HTML_PAGE)
        self.assertIn("class=\"disclosure\"", web.HTML_PAGE)
        self.assertIn("Show sessions", web.HTML_PAGE)
        self.assertNotIn('>${expanded ? "-" : "+"}</button>', web.HTML_PAGE)
        self.assertIn("input.indeterminate", web.HTML_PAGE)
        self.assertIn("state.selected.add(session.id)", web.HTML_PAGE)
        self.assertIn("Select shown codebases", web.HTML_PAGE)
        self.assertIn("Clear selection", web.HTML_PAGE)
        self.assertIn("for (const project of filteredProjects())", web.HTML_PAGE)

    def test_preview_filter_uses_local_minimizer_and_privacy_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            filter_command = make_fake_privacy_filter(root)

            result = web.preview_filter_text(
                text=f"Alice set OPENAI_API_KEY=demo-secret at {project}/.env and emailed alice@example.com.",
                project_path=str(project),
                privacy_filter_command=filter_command,
            )

            self.assertIn("<PROJECT>", result["minimized_text"])
            self.assertIn("<SECRET>", result["minimized_text"])
            self.assertIn("<PRIVATE_PERSON>", result["redacted_text"])
            self.assertIn("<PRIVATE_EMAIL>", result["redacted_text"])
            self.assertNotIn(str(project), result["redacted_text"])
            self.assertEqual(result["span_count"], 2)
            self.assertEqual(result["local_replacements"]["project_paths"], 1)
            self.assertEqual(result["local_replacements"]["secret_patterns"], 1)

    def test_package_selected_sessions_creates_zip_and_verifies_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            (project / ".git").mkdir(parents=True)
            codex_home = root / "codex"
            output_root = root / "exports"
            filter_command = make_fake_privacy_filter(root)

            write_jsonl(
                codex_home / "sessions/2026/05/08/rollout-web-private.jsonl",
                [
                    {
                        "type": "session_meta",
                        "timestamp": "2026-05-08T10:00:00Z",
                        "payload": {"id": "web-private", "cwd": str(project)},
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
                index = web.build_project_index({"codex"}, include_tools=False)
                selected = list(index["sessions"].values())
                result = web.package_selected_sessions(
                    selected=selected,
                    output_root=output_root,
                    privacy_filter_command=filter_command,
                    include_tools=False,
                )

            output_dir = Path(result["output_dir"])
            self.assertTrue(Path(result["zip_path"]).exists())
            self.assertTrue((output_dir / donation.SHAREABLE_DONATION_FILE).exists())
            self.assertTrue((output_dir / donation.SHAREABLE_MANIFEST_FILE).exists())
            self.assertTrue((output_dir / donation.REVIEW_FILE).exists())

            donation_text = (output_dir / donation.SHAREABLE_DONATION_FILE).read_text(encoding="utf-8")
            self.assertIn("<PRIVATE_PERSON>", donation_text)
            self.assertIn("<PRIVATE_EMAIL>", donation_text)
            self.assertNotIn(str(project), donation_text)

            verify_code = donation.command_verify(argparse.Namespace(output_dir=str(output_dir)))
            self.assertEqual(verify_code, 0)

            manifest = json.loads((output_dir / donation.SHAREABLE_MANIFEST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(manifest["stats"]["sessions"], 1)
            self.assertGreater(manifest["stats"]["estimated_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
