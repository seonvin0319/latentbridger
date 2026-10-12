#!/usr/bin/env bash
# understood as: train only Scene GSDTRL seeds 1 then 2 to exactly 1_000_000
# updates from the seed0 code and config, then compare six 1M h5 checkpoints
# with one shared evaluation seed. No algorithm changes, no extra seeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PIN="$ROOT/ops/scene_gsdtrl_seed12.json"
OUT="$ROOT/exp/scene-gsdtrl-seed12-20261013"
mkdir -p "$OUT"
exec 9>"$OUT/runner.lock"
if ! flock -n 9; then
  echo "runner already holds $OUT/runner.lock" >&2
  exit 1
fi

CURRENT_SEED=""
write_failure() {
  local rc="$1"
  local line="$2"
  python3 - "$OUT/failure.json" "$rc" "$line" "${CURRENT_SEED}" <<'PY'
import json, sys
from datetime import datetime, timezone
path, rc, line, seed = sys.argv[1:]
payload = {
    "status": "failed",
    "returncode": int(rc),
    "line": int(line),
    "seed": None if seed == "" else int(seed),
    "time": datetime.now(timezone.utc).isoformat(),
}
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
PY
}
trap 'rc=$?; write_failure "$rc" "$LINENO"; exit "$rc"' ERR

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT"
unset OGBENCH_DATASET_DIR || true

PY="$(python3 - "$PIN" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["python"])
PY
)"

"$PY" - "$PIN" "$ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
pin_path, root = sys.argv[1:]
pin = json.loads(Path(pin_path).read_text())
if pin["steps"] != 1_000_000 or pin["seeds"] != [1, 2]:
    raise SystemExit("pin is not the two-seed 1M Scene run")
if pin["dataset_dir_cli"] != "" or pin["batch_size"] != 1024 or pin["episodes"] != 50:
    raise SystemExit("pin changes a seed0 training setting")
for item in pin["datasets"]:
    path = Path(item["path"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != item["sha256"] or len(data) != item["bytes"]:
        raise SystemExit(f"dataset hash mismatch: {path}")
sys.path.insert(0, root)
from main_ctd_pathbridger import config_for
live = json.loads(json.dumps(config_for(pin["env"], pin["variant"]).to_dict()))
if live != pin["agent"]:
    raise SystemExit("live Scene GSDTRL config does not match the seed0 pin")
print("preflight ok", pin["base_commit"])
PY

gpu_uuid() {
  nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | head -1
}

wait_for_gpu() {
  local need=18000
  while true; do
    local row free uuid
    row="$(gpu_uuid)"
    uuid="${row%%,*}"
    uuid="${uuid// /}"
    free="${row##*,}"
    free="${free// /}"
    echo "gpu uuid=$uuid free_mib=$free need_mib=$need" >&2
    if [[ "$free" -ge "$need" ]]; then
      echo "$uuid"
      return 0
    fi
    sleep 30
  done
}

train_seed() {
  local seed="$1"
  CURRENT_SEED="$seed"
  local run_dir="$OUT/gsdtrl_weighted/seed${seed}"
  if [[ -e "$run_dir/complete.json" || -e "$run_dir/checkpoints" ]]; then
    echo "refusing to overwrite or resume $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  local uuid
  uuid="$(wait_for_gpu)"
  export JAX_PLATFORMS=cuda
  export CUDA_VISIBLE_DEVICES="$uuid"
  "$PY" - "$run_dir/launch.json" "$seed" "$PIN" "$(git rev-parse HEAD)" "$uuid" <<'PY'
import json, sys
from datetime import datetime, timezone
path, seed, pin_path, commit, uuid = sys.argv[1:]
pin = json.load(open(pin_path))
payload = {
    "time": datetime.now(timezone.utc).isoformat(),
    "commit": commit,
    "seed": int(seed),
    "steps": pin["steps"],
    "argv_matches_seed0_except_seed_and_run_dir": True,
    "cuda_visible_devices": uuid,
    "datasets": pin["datasets"],
}
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
PY
  set -o pipefail
  "$PY" main_ctd_pathbridger.py \
    --env scene \
    --variant gsdtrl_weighted \
    --seed "$seed" \
    --steps 1000000 \
    --episodes 50 \
    --run-dir "$run_dir" \
    2>&1 | tee -a "$run_dir/run.log"
  "$PY" - "$run_dir" "$seed" <<'PY'
import json, sys
from pathlib import Path
run_dir, seed = Path(sys.argv[1]), int(sys.argv[2])
record_path = run_dir / "complete.json"
record = json.loads(record_path.read_text())
if record.get("steps") != 1_000_000 or record.get("smoke") or record.get("variant") != "gsdtrl_weighted":
    record_path.rename(run_dir / "complete.rejected.json")
    raise SystemExit(f"seed {seed} is not a finished 1M GSDTRL run: {record}")
eval_path = run_dir / "evaluation_1000000_h5.json"
row = json.loads(eval_path.read_text())
expected = (1_000_000, 5, 5, 50, seed, "gsdtrl_weighted")
got = (row.get("checkpoint"), row.get("h"), row.get("num_tasks"), row.get("episodes_per_task"), row.get("seed"), row.get("variant"))
if got != expected:
    raise SystemExit(f"seed {seed} h5 metadata {got} != {expected}")
ckpt = run_dir / "checkpoints" / "params_1000000.pkl"
if not ckpt.is_file():
    raise SystemExit(f"missing {ckpt}")
print(f"seed {seed} complete h5={row['overall_success']}")
PY
}

train_seed 1
train_seed 2
CURRENT_SEED=""
uuid="$(wait_for_gpu)"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES="$uuid"
"$PY" ops/eval_scene_common_h5.py --config "$PIN" --output "$OUT/common_h5"
echo "scene gsdtrl seed1 and seed2 finished at 1M; common h5 table is $OUT/common_h5"
