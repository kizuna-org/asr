# 2024-07-01 env.template commentout fix

## Summary

Commented out all non-KEY=value lines (YAML front matter and Markdown documentation) in `infra/poc/frp-client/env.template` to prevent parsing errors during docker compose execution. This ensures only valid KEY=value pairs and comments remain active in the template.

## Details
- Lines 1-41 (YAML front matter and Markdown) are now prefixed with `#`.
- Only KEY=value lines and existing comments are left unchanged.
- This prevents accidental copying of invalid content into production `.env` files and avoids docker compose errors.

## Motivation
Previously, the env.template included documentation and YAML front matter, which are not valid in `.env` files. Docker compose would fail to parse these lines, causing errors. By commenting them out, the template is now safe to copy directly to `.env`.

---

作業日: 2024-07-01
担当: AI 
