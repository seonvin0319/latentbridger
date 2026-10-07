#!/usr/bin/env bash
# Secondary waiter for the seed-0 CTD suite.
# It only reads CPB artifacts/process state and never signals existing jobs.
set -u
ROOT=/home/shchoi/latentbridger_ctd
CPB=/home/shchoi/latentbridger/exp/contrastive_pathbridger
PY=/home/shchoi/latentbridger/.venv/bin/python
mkdir -p "$ROOT/exp/ctd_pathbridger"
exec >>"$ROOT/exp/ctd_pathbridger/waiter.log" 2>&1
echo "ctd waiter start $(date --iso-8601=seconds)"
echo "method order: CTD-W, DTRL-W, CTD-PathNCE-W, CTD-U, DTRL-U, CTD-PathNCE-U, CTD-PathNCE-W+BridgeGeo, CTD-PathNCE-U+BridgeGeo"

queue_ready() {
  "$PY" - "$CPB" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
needed = []
for env in ("puzzle_3x3", "antmaze_medium"):
    run = root / env / "cpb_rank_only" / "seed0"
    complete = run / "complete.json"
    if not complete.exists():
        needed.append(str(complete))
        continue
    record = json.loads(complete.read_text())
    if int(record.get("steps", 0)) != 1_000_000:
        needed.append(f"{complete} steps={record.get('steps')}")
    for name in (
        "evaluation_1000000_h1.json",
        "evaluation_1000000_h2.json",
        "evaluation_1000000_h5.json",
        "checkpoints/params_1000000.pkl",
    ):
        if not (run / name).exists():
            needed.append(str(run / name))
live = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        continue
    if "run_contrastive_pathbridger_suite.py" in command or "main_contrastive_pathbridger.py" in command:
        live.append(entry.name)
if needed or live:
    print("waiting:", "; ".join(needed + [f"pid {pid}" for pid in live]))
    sys.exit(1)
sys.exit(0)
PY
}

while ! queue_ready; do
  sleep 60
done
echo "cpb queue complete $(date --iso-8601=seconds)"
# The report must reach the user before a GPU process starts.
REPORT_GRACE_SECONDS=${REPORT_GRACE_SECONDS:-300}
echo "report grace ${REPORT_GRACE_SECONDS}s before CTD launch"
sleep "$REPORT_GRACE_SECONDS"
if ! queue_ready; then
  echo "CPB became active again during report grace; returning to wait"
  while ! queue_ready; do
    sleep 60
  done
fi
echo "starting seed0 CTD suite $(date --iso-8601=seconds)"
cd "$ROOT"
exec "$PY" -u scripts/run_ctd_suite.py
