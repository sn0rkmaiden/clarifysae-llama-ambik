#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
This legacy helper was ambiguous and referenced a missing config.
For AmbiK v1, build and run a manifest:
  ./scripts/build_ambik_v1_manifests.sh
  ./scripts/run_manifest_row.sh manifests/generated/ambik_explore_v1.csv ROW_INDEX
EOF
exit 2
