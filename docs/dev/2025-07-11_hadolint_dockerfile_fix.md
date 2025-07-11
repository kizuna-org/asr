# 2025-07-11 Dockerfile hadolint Compliance and Version Pinning

## Summary
- Updated `infra/poc/frp-client/Dockerfile` and `infra/poc/frp-client/jenkins/Dockerfile` to pass `hadolint`.
- Pinned all apt and base image versions explicitly.
- Used `--no-install-recommends` and removed apt lists after install.
- Consolidated multiple `RUN` instructions in Jenkins Dockerfile.
- Applied lll (line length limit) and linting best practices.
- Committed changes as per project rules.

## Details
- `FROM ubuntu/squid:latest` → `FROM ubuntu/squid:6.6-24.04`
- `apt-get install` now pins:
  - `busybox=1.30.1-4ubuntu6.5`
  - `tcpdump=4.9.3-4ubuntu0.3`
  - Jenkins Dockerfile pins all system packages and Docker CLI.
- All warnings and info from hadolint resolved.

## Author
Automated by AI (Cursor) 
