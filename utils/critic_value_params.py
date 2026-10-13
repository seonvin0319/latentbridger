"""Original critic value-parameter selection used by legacy main.py.

Source of truth: PathBridger_legacy
``3999f3759fe6bfce1e62966bb38f0ce277add843`` ``main.py::_extract_critic_value_params``.

Do not simplify this to always pass the target or always pass the online value.
The original CriticAgent has no ``_is_trl``, so the online ``modules_value`` branch
is taken for PathBridger training; the TRL hasattr branch is retained verbatim.
"""
from __future__ import annotations

from typing import Any


def extract_critic_value_params(critic_agent: Any) -> Any | None:
    if critic_agent is None:
        return None
    if hasattr(critic_agent, '_is_trl') and bool(critic_agent._is_trl()):
        return critic_agent.network.params.get('modules_target_value', None)
    return critic_agent.network.params.get('modules_value', None)
