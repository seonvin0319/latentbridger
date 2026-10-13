"""Load the unmodified legacy helper by AST extraction (no rewritten selection)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

LEGACY_MAIN = Path('/home/choi/pb_reimpl_workspace/PathBridger_legacy/main.py')
HELPER_NAME = '_extract_critic_value_params'


def extract_helper_source(legacy_main: Path = LEGACY_MAIN) -> str:
    source = legacy_main.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER_NAME:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise RuntimeError(f'AST source segment missing for {HELPER_NAME}')
            return text
    raise RuntimeError(f'{HELPER_NAME} not found in {legacy_main}')


def load_extract_critic_value_params(legacy_main: Path = LEGACY_MAIN):
    """Return the original function object, executed from the extracted source."""
    text = extract_helper_source(legacy_main)
    # The original annotation uses Any; provide it in the exec globals.
    namespace: dict[str, Any] = {'Any': Any}
    exec(text, namespace)
    fn = namespace[HELPER_NAME]
    # Guard: source must still be the known branch, not a rewritten stub.
    assert 'modules_target_value' in text and 'modules_value' in text
    assert '_is_trl' in text
    return fn, text
