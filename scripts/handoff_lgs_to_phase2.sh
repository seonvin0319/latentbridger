#!/usr/bin/env bash
# Wait for archival puzzle LGS_TRL_W_FROZEN 1M completion, then stop the old
# FutureNCE queue (so it does not start LGSDTRL/cube) and launch phase-2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_UNIT="${OLD_UNIT:-learned-goalspace-svcho-20261009-232503.service}"
LGS_DIR="$ROOT/exp/learned_goalspace/downstream/puzzle_3x3/LGS_TRL_W_FROZEN/seed0"
PYTHON_BIN="${PYTHON:-/home/svcho/anaconda3/envs/offrl/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-30}"

lgs_complete() {
  [[ -f "$LGS_DIR/complete.json" ]] || return 1
  [[ -f "$LGS_DIR/checkpoints/params_1000000.pkl" ]] || return 1
  python3 - "$LGS_DIR/complete.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["steps"]) == 1_000_000)
PY
}

echo "[handoff] watching $LGS_DIR (old unit=$OLD_UNIT)"
while true; do
  if complete="$(lgs_complete)" && [[ "$complete" == "True" ]]; then
    echo "[handoff] archival LGS_TRL_W_FROZEN complete at $(date '+%F %T %Z')"
    break
  fi
  if ! systemctl --user is-active --quiet "$OLD_UNIT"; then
    if complete="$(lgs_complete)" && [[ "$complete" == "True" ]]; then
      echo "[handoff] old unit already inactive and LGS complete"
      break
    fi
    echo "[handoff] old unit inactive but LGS incomplete; refusing phase-2" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

if systemctl --user is-active --quiet "$OLD_UNIT"; then
  echo "[handoff] stopping old FutureNCE queue to avoid LGSDTRL/cube"
  systemctl --user stop "$OLD_UNIT"
fi

echo "[handoff] launching phase-2 under 16GiB cgroup"
cd "$ROOT"
exec env PYTHON="$PYTHON_BIN" "$ROOT/scripts/run_learned_goalspace_phase2_16g.sh" "$@"
