#!/usr/bin/env bash
# Oracle PB only, seed 0, 1M — no smoke. Up to PARALLEL jobs on one GPU (default 2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/home/shchoi/oracle-pb-on-pb-e592665-20261014}"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/runs" "${OUT_ROOT}/manifests"

COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
COMMIT_DIRTY="$(git -C "${ROOT}" status --porcelain | wc -l | tr -d ' ')"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
DATASET_DIR="${DATASET_DIR:-/home/shchoi/.ogbench/data}"
PARALLEL="${PARALLEL:-2}"

# Fixed order — do not reorder without updating the handoff note.
ENVS=(
  puzzle_3x3
  puzzle_4x4
  cube_single
  cube_double
  cube_triple
  antmaze_medium
  antmaze_large
  scene
)

cat > "${OUT_ROOT}/manifests/suite_provenance.json" <<EOF
{
  "started_at": "$(date --iso-8601=seconds)",
  "repo": "${ROOT}",
  "commit": "${COMMIT}",
  "working_tree_dirty_paths": ${COMMIT_DIRTY},
  "backbone_pin": "e592665923973655696889d0882c245bc7d2b314",
  "train_steps": ${TRAIN_STEPS},
  "eval_episodes": ${EVAL_EPISODES},
  "eval_interval": ${EVAL_INTERVAL},
  "save_interval": ${SAVE_INTERVAL},
  "dataset_dir": "${DATASET_DIR}",
  "parallel": ${PARALLEL},
  "modes": ["oracle_pb"],
  "seeds": [0],
  "smoke": false,
  "env_order": ["puzzle_3x3","puzzle_4x4","cube_single","cube_double","cube_triple","antmaze_medium","antmaze_large","scene"],
  "note": "Oracle PB seed0 1M only; no PB baseline; no smoke; 2-way parallel"
}
EOF

need_free_mib="${NEED_FREE_MIB:-12000}"
wait_for_gpu() {
  echo "[$(date --iso-8601=seconds)] waiting for GPU (foreign trainers outside OUT_ROOT=0)..."
  while true; do
    foreign=$(ps -eo pid,cmd | grep -E '[.]venv/bin/python.*(main\.py|run_offline)' | grep -v grep | grep -v "${OUT_ROOT}" | wc -l || true)
    echo "[$(date --iso-8601=seconds)] foreign_trainers=${foreign}"
    if [[ "${foreign}" -eq 0 ]]; then
      break
    fi
    sleep 30
  done
}

JOBS_FILE="${OUT_ROOT}/manifests/job_queue.txt"
: > "${JOBS_FILE}"
for cfg in "${ENVS[@]}"; do
  echo "oracle_pb|${cfg}|0|${TRAIN_STEPS}|oracle_pb_${cfg}" >> "${JOBS_FILE}"
done
cp "${JOBS_FILE}" "${OUT_ROOT}/manifests/job_queue_snapshot.txt"

is_tag_running() {
  local tag="$1"
  ps -eo args | grep -F "/runs/${tag}/seed" | grep -F 'main.py' | grep -v grep >/dev/null 2>&1
}

run_one() {
  local mode="$1" cfg="$2" seed="$3" steps="$4" tag="$5"
  local save_dir="${OUT_ROOT}/runs/${tag}/seed${seed}"
  mkdir -p "${save_dir}"
  if [[ -f "${save_dir}/DONE" ]]; then
    echo "skip finished ${save_dir}"
    return 0
  fi
  if is_tag_running "${tag}"; then
    echo "skip already-running ${tag}"
    while is_tag_running "${tag}"; do sleep 30; done
    [[ -f "${save_dir}/DONE" ]] && return 0
    echo "existing job exited without DONE: ${save_dir}" >&2
    return 1
  fi
  local agent_cfg="${ROOT}/configs/${mode}/${cfg}.py"
  local started_at
  started_at="$(date --iso-8601=seconds)"
  cat > "${save_dir}/run_manifest.json" <<EOF
{
  "tag": "${tag}",
  "mode_dir": "${mode}",
  "config_file": "${agent_cfg}",
  "seed": ${seed},
  "train_steps": ${steps},
  "eval_episodes": ${EVAL_EPISODES},
  "eval_interval": ${EVAL_INTERVAL},
  "save_interval": ${SAVE_INTERVAL},
  "dataset_dir": "${DATASET_DIR}",
  "commit": "${COMMIT}",
  "started_at": "${started_at}"
}
EOF
  echo "[${started_at}] START ${tag} steps=${steps}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${PY}" -u "${ROOT}/main.py" \
    --agent="${agent_cfg}" \
    --seed="${seed}" \
    --train_steps="${steps}" \
    --log_interval=5000 \
    --eval_interval="${EVAL_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --eval_episodes="${EVAL_EPISODES}" \
    --eval_final=True \
    --use_wandb=False \
    --dataset_dir="${DATASET_DIR}" \
    --save_dir="${save_dir}" \
    > "${OUT_ROOT}/logs/${tag}_seed${seed}.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    touch "${save_dir}/DONE"
    echo "[$(date --iso-8601=seconds)] DONE ${save_dir}"
  else
    echo "[$(date --iso-8601=seconds)] FAIL rc=${rc} ${save_dir}" | tee -a "${OUT_ROOT}/logs/suite_stdout.log"
    return "$rc"
  fi
}

wait_for_gpu
echo "[$(date --iso-8601=seconds)] Oracle-PB 1M queue PARALLEL=${PARALLEL} (no smoke)" | tee -a "${OUT_ROOT}/logs/suite_stdout.log"
cat "${JOBS_FILE}" | tee -a "${OUT_ROOT}/logs/suite_stdout.log"

running=0
declare -A PID_TAG
fail=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" ]] && continue
  IFS='|' read -r mode cfg seed steps tag <<<"${line}"
  save_dir="${OUT_ROOT}/runs/${tag}/seed${seed}"
  if [[ -f "${save_dir}/DONE" ]]; then
    echo "skip finished ${save_dir}"
    continue
  fi
  while [[ "${running}" -ge "${PARALLEL}" ]]; do
    for pid in "${!PID_TAG[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || fail=1
        echo "[$(date --iso-8601=seconds)] slot free (was ${PID_TAG[$pid]})"
        unset 'PID_TAG[$pid]'
        running=$((running - 1))
      fi
    done
    if [[ "${running}" -ge "${PARALLEL}" ]]; then
      sleep 30
    fi
  done
  run_one "${mode}" "${cfg}" "${seed}" "${steps}" "${tag}" &
  pid=$!
  PID_TAG[$pid]="${tag}/seed${seed}"
  running=$((running + 1))
  sleep 10
done < "${JOBS_FILE}"

for pid in "${!PID_TAG[@]}"; do
  wait "${pid}" || fail=1
done
echo "[$(date --iso-8601=seconds)] suite finished fail=${fail}" | tee -a "${OUT_ROOT}/logs/suite_stdout.log"
exit "${fail}"
