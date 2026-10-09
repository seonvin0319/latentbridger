"""Intention-Conditioned PathBridger (see INTENTION_PATHBRIDGER.md).

The seed-0 PB checkpoints in ``checkpoints/1m_env_best`` were produced by commit ``21a4042``; the current
working tree changed the PB module definitions (e.g. removed the sinusoidal flow-time embedding), so the
checkpoints cannot be loaded with it. All PB code (agents/, utils/, main.py) is therefore imported from an
unmodified export of that commit, ``IPB_PB_CODE_DIR`` (default ``../Pathbridger_flow_pb21a4042``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PB_CODE_COMMIT = '21a4042'
PB_CODE_DIR = Path(os.environ.get('IPB_PB_CODE_DIR', Path(__file__).resolve().parents[2] / 'Pathbridger_flow_pb21a4042')).resolve()


def _install_pb_code_path() -> None:
    marker = PB_CODE_DIR / 'PB_CODE_COMMIT'
    if not (PB_CODE_DIR / 'agents' / 'dynamics.py').is_file() or not marker.is_file():
        raise FileNotFoundError(
            f'PB code export missing at {PB_CODE_DIR}. Create it with: '
            f'git archive {PB_CODE_COMMIT} | tar -x -C {PB_CODE_DIR} && echo {PB_CODE_COMMIT} > {marker}'
        )
    if marker.read_text().strip() != PB_CODE_COMMIT:
        raise RuntimeError(f'{marker} does not match expected commit {PB_CODE_COMMIT}')
    for name in ('agents', 'utils', 'main', 'eval_checkpoint', 'rollout'):
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, '__file__', '') or '').startswith(str(PB_CODE_DIR)):
            raise ImportError(f'Module {name!r} already imported from {mod.__file__}; import intention_pb first.')
    if sys.path[:1] != [str(PB_CODE_DIR)]:
        sys.path[:] = [str(PB_CODE_DIR)] + [p for p in sys.path if p != str(PB_CODE_DIR)]


ensure_pb_code_path = _install_pb_code_path
_install_pb_code_path()
