#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
The old generic strength sweep referenced a missing config and is disabled.
Use configs/experiments/ambik_explore_v1.yaml and build_manifest instead.
EOF
exit 2
