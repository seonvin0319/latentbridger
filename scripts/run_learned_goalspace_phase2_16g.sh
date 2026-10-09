#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT=17179869184
HIGH=16106127360
PYTHON_BIN="${PYTHON:-$(command -v python)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Refusing launch: Python executable is unavailable: $PYTHON_BIN" >&2
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

if [[ "${LGS_16G_REEXEC:-0}" == "1" ]]; then
  if ! verify_cgroup_contract; then
    echo "Refusing launch: the full 16 GiB cgroup safety contract is not active." >&2
    exit 1
  fi
  cd "$ROOT"
  exec "$PYTHON_BIN" scripts/run_learned_goalspace_phase2_queue.py "$@"
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "Refusing launch: systemd-run is unavailable." >&2
  exit 1
fi

unit="learned-goalspace-phase2-${USER}-$(date +%Y%m%d-%H%M%S)"
exec systemd-run --user --wait --collect \
  --unit="$unit" \
  --working-directory="$ROOT" \
  --property=MemoryHigh=15G \
  --property=MemoryMax=16G \
  --property=MemorySwapMax=0 \
  --property=OOMPolicy=kill \
  --property=KillMode=control-group \
  --setenv=LGS_16G_REEXEC=1 \
  --setenv=PYTHON="$PYTHON_BIN" \
  --setenv=PATH="$PATH" \
  "$0" "$@"
