#!/usr/bin/env python3
"""
OPF-compatible JSON wrapper for OpenMed MLX Privacy Filter artifacts.

The donation exporter only needs a command that accepts text and emits JSON with
`redacted_text` plus optional `detected_spans`. This script adapts OpenMed's
direct MLX pipeline to that shape so Apple Silicon machines can use MLX without
changing the exporter contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OPENMED_MLX_MODEL = "OpenMed/privacy-filter-mlx-8bit"


PLACEHOLDER_BY_LABEL = {
    "account_number": "<ACCOUNT_NUMBER>",
    "private_address": "<PRIVATE_ADDRESS>",
    "private_date": "<PRIVATE_DATE>",
    "private_email": "<PRIVATE_EMAIL>",
    "private_person": "<PRIVATE_PERSON>",
    "private_phone": "<PRIVATE_PHONE>",
    "private_url": "<PRIVATE_URL>",
    "secret": "<SECRET>",
    "address": "<PRIVATE_ADDRESS>",
    "api_key": "<SECRET>",
    "date": "<PRIVATE_DATE>",
    "date_of_birth": "<PRIVATE_DATE>",
    "email": "<PRIVATE_EMAIL>",
    "first_name": "<PRIVATE_PERSON>",
    "last_name": "<PRIVATE_PERSON>",
    "name": "<PRIVATE_PERSON>",
    "password": "<SECRET>",
    "phone": "<PRIVATE_PHONE>",
    "phone_number": "<PRIVATE_PHONE>",
    "token": "<SECRET>",
    "url": "<PRIVATE_URL>",
}


def placeholder_for_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith(("b_", "i_", "e_", "s_")):
        normalized = normalized[2:]
    return PLACEHOLDER_BY_LABEL.get(normalized, f"<{normalized.upper()}>")


def span_label(entity: dict[str, Any]) -> str:
    label = entity.get("entity_group") or entity.get("label") or entity.get("entity")
    return str(label or "privacy")


def normalize_entities(entities: Iterable[Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            entity = entity.__dict__ if hasattr(entity, "__dict__") else {}
        label = span_label(entity)
        try:
            start = int(entity["start"])
            end = int(entity["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        span: dict[str, Any] = {
            "label": label,
            "start": start,
            "end": end,
            "placeholder": placeholder_for_label(label),
        }
        score = entity.get("score") or entity.get("confidence")
        if score is not None:
            try:
                span["score"] = float(score)
            except (TypeError, ValueError):
                pass
        spans.append(span)
    return sorted(spans, key=lambda item: (item["start"], item["end"]))


def redact_text(text: str, spans: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        pieces.append(str(span["placeholder"]))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return

    try:
        mx.clear_cache()
    except Exception:
        return


class OpenMedMLXFilter:
    def __init__(self, model_name: str = DEFAULT_OPENMED_MLX_MODEL):
        try:
            from huggingface_hub import snapshot_download
            from openmed.mlx.inference import create_mlx_pipeline
        except ImportError as exc:
            raise RuntimeError(
                'OpenMed MLX dependencies are missing. Install with `uv pip install "openmed[mlx]"`.'
            ) from exc

        model_path = snapshot_download(model_name)
        self.pipeline = create_mlx_pipeline(model_path)

    def filter_text(self, text: str) -> dict[str, Any]:
        try:
            spans = normalize_entities(self.pipeline(text))
        finally:
            clear_mlx_cache()
        redacted = redact_text(text, spans)
        return {
            "schema_version": 1,
            "summary": {"span_count": len(spans)},
            "detected_spans": spans,
            "redacted_text": redacted,
        }


def load_openmed_entities(text: str, model_name: str) -> list[dict[str, Any]]:
    return normalize_entities(OpenMedMLXFilter(model_name).pipeline(text))


def filter_text(text: str, model_name: str) -> dict[str, Any]:
    return OpenMedMLXFilter(model_name).filter_text(text)


def iter_inputs(args: argparse.Namespace) -> Iterable[str]:
    if args.text:
        yield args.text
        return
    if args.text_file:
        for raw_path in args.text_file:
            path = Path(raw_path).expanduser()
            text = path.read_text(encoding="utf-8")
            if text:
                yield text
        return
    if not sys.stdin.isatty():
        text = sys.stdin.read()
        if text:
            yield text
        return
    raise RuntimeError("provide text, --text-file, or stdin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenMed MLX Privacy Filter with OPF-compatible JSON output.")
    parser.add_argument("text", nargs="?", help="text to filter")
    parser.add_argument("-f", "--text-file", action="append", help="text file to filter; repeatable")
    parser.add_argument("--model", default=DEFAULT_OPENMED_MLX_MODEL, help="OpenMed MLX model id or local snapshot path")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="output format")
    parser.add_argument("--json-indent", type=int, default=0, help="JSON indentation; 0 means compact")
    parser.add_argument("--output-mode", default="typed", help="accepted for OPF CLI compatibility")
    parser.add_argument("--no-print-color-coded-text", action="store_true", help="accepted for OPF CLI compatibility")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        redactor = OpenMedMLXFilter(args.model)
        for text in iter_inputs(args):
            result = redactor.filter_text(text)
            if args.format == "text":
                print(result["redacted_text"])
            else:
                indent = None if args.json_indent == 0 else args.json_indent
                print(json.dumps(result, ensure_ascii=False, indent=indent))
        return 0
    except Exception as exc:
        print(f"OpenMed MLX privacy filter failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
