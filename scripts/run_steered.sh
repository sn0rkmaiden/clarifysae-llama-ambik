#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
This legacy helper referenced a missing config and is intentionally disabled.
Use the manifest-driven workflow in docs/EXPERIMENTS_V1.md.
EOF
exit 2
