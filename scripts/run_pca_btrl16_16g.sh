#!/usr/bin/env bash
# Detached 16 GiB cgroup launch for NonOracle PCA-BTRL16 puzzle seed0 1M.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT=17179869184
HIGH=16106127360
PYTHON_BIN="${PYTHON:-/home/svcho/anaconda3/envs/offrl/bin/python}"
OFFRL_ROOT="$(dirname "$(dirname "$PYTHON_BIN")")"
NVIDIA_LIBS="$(find "$OFFRL_ROOT"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/usr/local/cuda/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

ENV_NAME="${ENV_NAME:-puzzle_3x3}"
METHOD="${METHOD:-PCA_BTRL16}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-$ROOT/exp/nonoracle_goalspace/pca_btrl16/${ENV_NAME}/seed${SEED}}"
PCA_PATH="${PCA_PATH:-$ROOT/exp/learned_goalspace/pca16/${ENV_NAME}/seed${SEED}/representation.pkl}"
mkdir -p "$RUN_DIR" "$ROOT/exp/nonoracle_goalspace/logs"

# Never touch BTRL16 outputs.
BTRL_COMPLETE="$ROOT/exp/nonoracle_goalspace/btrl16/${ENV_NAME}/seed${SEED}/complete.json"
if [[ ! -f "$BTRL_COMPLETE" ]]; then
  echo "Refusing launch: expected preserved BTRL16 complete at $BTRL_COMPLETE" >&2
  exit 1
fi
if [[ ! -f "$PCA_PATH" ]]; then
  echo "Refusing launch: missing TRAIN-only PCA16 at $PCA_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Refusing launch: Python unavailable: $PYTHON_BIN" >&2
  exit 1
fi

current_cgroup_file() {
  local name="$1" path relative
  relative="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
  path="/sys/fs/cgroup${relative%/}/${name}"
  [[ -r "$path" ]] || path="/sys/fs/cgroup/${name}"
  [[ -r "$path" ]] || return 1
  printf '%s\n' "$path"
}

verify_cgroup_contract() {
  local max_path high_path swap_path oom_path max high swap oom
  max_path="$(current_cgroup_file memory.max)" || return 1
  high_path="$(current_cgroup_file memory.high)" || return 1
  swap_path="$(current_cgroup_file memory.swap.max)" || return 1
  oom_path="$(current_cgroup_file memory.oom.group)" || return 1
  max="$(<"$max_path")"
  high="$(<"$high_path")"
  swap="$(<"$swap_path")"
  oom="$(<"$oom_path")"
  [[ "$max" =~ ^[0-9]+$ ]] && (( max <= LIMIT )) || return 1
  [[ "$high" =~ ^[0-9]+$ ]] && (( high <= HIGH )) || return 1
  [[ "$swap" == "0" && "$oom" == "1" ]]
}

if [[ "${PCA_BTRL_16G_REEXEC:-0}" == "1" ]]; then
  if ! verify_cgroup_contract; then
    echo "Refusing launch: 16 GiB cgroup contract not active." >&2
    exit 1
  fi
  cd "$ROOT"
  "$PYTHON_BIN" -c 'import jax; assert any("cuda" in str(d).lower() for d in jax.devices()), jax.devices()'
  exec "$PYTHON_BIN" main_pca_btrl16.py \
    --env "$ENV_NAME" \
    --method "$METHOD" \
    --seed "$SEED" \
    --steps 1000000 \
    --batch-size 1024 \
    --episodes 50 \
    --pca-path "$PCA_PATH" \
    --run-dir "$RUN_DIR" \
    ${RESUME:+--resume}
fi

unit="pca-btrl16-${USER}-$(date +%Y%m%d-%H%M%S)"
systemd-run --user --collect \
  --unit="$unit" \
  --working-directory="$ROOT" \
  --property=MemoryHigh=15G \
  --property=MemoryMax=16G \
  --property=MemorySwapMax=0 \
  --property=OOMPolicy=kill \
  --property=KillMode=control-group \
  --setenv=PCA_BTRL_16G_REEXEC=1 \
  --setenv=PYTHON="$PYTHON_BIN" \
  --setenv=PATH="$PATH" \
  --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --setenv=MUJOCO_GL=egl \
  --setenv=ENV_NAME="$ENV_NAME" \
  --setenv=METHOD="$METHOD" \
  --setenv=SEED="$SEED" \
  --setenv=RUN_DIR="$RUN_DIR" \
  --setenv=PCA_PATH="$PCA_PATH" \
  --setenv=RESUME="${RESUME:-}" \
  "$0" "$@"

echo "Started unit=$unit run_dir=$RUN_DIR"
echo "$unit" > "$ROOT/exp/nonoracle_goalspace/logs/pca_btrl16_unit.txt"
echo "$RUN_DIR" > "$ROOT/exp/nonoracle_goalspace/logs/pca_btrl16_run_dir.txt"
