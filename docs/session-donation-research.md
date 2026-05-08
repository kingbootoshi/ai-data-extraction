# Project-Scoped Session Donation Research

This fork adds a safer donation path for Codex and Claude Code sessions. The goal is to let a user donate only sessions tied to one project while avoiding accidental leakage from other projects, global history, or system-level records.

## Codex Storage

Codex stores local session transcripts as rollout JSONL files under:

```text
~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
```

The OpenAI Codex source shows rollouts are persisted as JSONL and that new session paths are created under `codex_home/sessions/YYYY/MM/DD`. Each new session writes a `session_meta` record containing the session id, timestamp, cwd, originator, version, source, model provider, and related metadata.

Project scoping is not encoded by directory. It is carried in session metadata, especially `session_meta.payload.cwd`, and in later turn context entries. The exporter therefore reads every discovered `cwd` value in a candidate session and only marks the session eligible when all cwd values are equal to, or descendants of, the selected project root.

Related sources:

- `openai/codex` rollout recorder: https://github.com/openai/codex/blob/main/codex-rs/rollout/src/recorder.rs
- Codex issue noting `session_meta.payload.cwd` enables cwd filtering: https://github.com/openai/codex/issues/6730
- Codex local files discussed in index/session issue: https://github.com/openai/codex/issues/20165

## Claude Code Storage

Claude Code stores local project transcripts as append-only JSONL files under:

```text
~/.claude/projects/<sanitized-project-path>/<session-id>.jsonl
```

Current records commonly include `type` values such as `user`, `assistant`, `system`, `summary`, `tool_result`, `permission-mode`, and `file-history-snapshot`. Message records carry fields such as `cwd`, `sessionId`, `timestamp`, `gitBranch`, and nested `message.content`.

The exporter does not trust the sanitized project directory name by itself. It parses the JSONL records and applies the same strict cwd rule used for Codex: every discovered cwd must remain inside the selected project root. System records are not exported.

Related sources:

- Claude Code schema feature discussion: https://github.com/anthropics/claude-code/issues/53516
- Community schema research: https://github.com/neilberkman/ccrider/blob/main/research/schema.md

## Privacy Filter

OpenAI Privacy Filter is a Hugging Face token-classification model for privacy span detection and masking. It detects these span categories:

- `account_number`
- `private_address`
- `private_email`
- `private_person`
- `private_phone`
- `private_url`
- `private_date`
- `secret`

The exporter requires an OpenAI Privacy Filter compatible command before it writes donation data. By default it calls:

```bash
opf --output-mode typed
```

The command must emit JSON with `redacted_text` and optional `detected_spans`, matching the documented OPF output shape. The exporter strips raw span text from saved metadata and keeps only labels, offsets, placeholders, and scores.

Related sources:

- Hugging Face model card: https://huggingface.co/openai/privacy-filter
- Privacy Filter repository: https://github.com/openai/privacy-filter
- OPF output schema: https://github.com/openai/privacy-filter/blob/main/OUTPUT_SCHEMAS.md

## Safety Rules Implemented

- Sessions are eligible only if every recorded cwd is inside the selected project root.
- Codex global files such as `history.jsonl`, `session_index.jsonl`, and SQLite state are not exported.
- Claude global history and system records are not exported.
- Shareable donation records contain session hashes, not raw source file paths.
- Message and optional tool text are minimized for local paths and common token patterns before being sent through OpenAI Privacy Filter.
- Donation output is refused if the privacy filter command is missing or fails.
- A manifest and review HTML are generated so users can inspect exactly what would be shared.
- The verifier checks the donation hash, record count, and per-message privacy-filter metadata.

