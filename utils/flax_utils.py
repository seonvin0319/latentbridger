"""Minimal Flax training-state and checkpoint utilities."""

from __future__ import annotations

import functools
import os
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import numpy as np
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """Expose a dictionary of Flax modules through one shared parameter tree."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name: str | None = None, **kwargs):
        if name is not None:
            return self.modules[name](*args, **kwargs)

        if kwargs.keys() != self.modules.keys():
            raise ValueError(
                'Initialization arguments must exactly match the module names: '
                f'expected {tuple(self.modules)}, got {tuple(kwargs)}.'
            )

        outputs = {}
        for module_name, module_args in kwargs.items():
            module = self.modules[module_name]
            if isinstance(module_args, Mapping):
                outputs[module_name] = module(**module_args)
            elif isinstance(module_args, Sequence):
                outputs[module_name] = module(*module_args)
            else:
                outputs[module_name] = module(module_args)
        return outputs


class TrainState(flax.struct.PyTreeNode):
    """Parameters, optimizer state, and the callable Flax module definition."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Any,
        tx: optax.GradientTransformation | None = None,
        **kwargs,
    ):
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=None if tx is None else tx.init(params),
            **kwargs,
        )

    def __call__(self, *args, params=None, method: str | None = None, **kwargs):
        variables = {'params': self.params if params is None else params}
        method_fn = None if method is None else getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method_fn, **kwargs)

    def select(self, name: str):
        """Return a callable bound to one module in a :class:`ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        if self.tx is None:
            raise ValueError('Cannot apply gradients without an optimizer.')
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=params, opt_state=opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn):
        """Differentiate ``loss_fn(params)`` and apply one optimizer update."""
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads), info


_CHECKPOINT_NAME = re.compile(r'^params_(\d+)\.pkl$')


def resolve_checkpoint(
    path: str | os.PathLike[str],
    step: int = 0,
) -> tuple[Path, int]:
    """Resolve a checkpoint path and its unambiguous training step.

    A zero step means "infer from an exact ``params_<step>.pkl`` filename".
    Checkpoint directories always require an explicit positive step.
    """

    path = Path(path)
    if path.suffix == '.pkl':
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        inferred_step = int(match.group(1)) if match is not None else None
        requested_step = int(step)
        if requested_step < 0:
            raise ValueError('Checkpoint step cannot be negative.')
        if requested_step == 0:
            if inferred_step is None:
                raise ValueError(
                    'An exact checkpoint with a nonstandard filename requires '
                    'an explicit positive step.'
                )
            requested_step = inferred_step
        elif inferred_step is not None and requested_step != inferred_step:
            raise ValueError(
                f'Checkpoint filename implies step {inferred_step}, but '
                f'step {requested_step} was requested.'
            )
        return path, requested_step

    step = int(step)
    if step < 1:
        raise ValueError(
            'A checkpoint directory requires an explicit positive step.'
        )
    return path / f'params_{step}.pkl', step


CHECKPOINT_FORMAT = 'pathbridger/3999f37/v1'


def save_agent(agent, save_dir, step):
    """Atomically save full original states, config, and the host RNG stream."""
    from agents.pathbridger import CORE_REVISION
    save_path, _ = resolve_checkpoint(save_dir, step)
    if int(agent.dynamics.network.step) != step + 1 or int(agent.critic.network.step) != step + 1:
        raise ValueError('Checkpoint step does not match both original optimizer states.')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': CHECKPOINT_FORMAT, 'core_revision': CORE_REVISION,
        'config': dict(agent.config),
        'obs_dim': int(agent.dynamics.network.model_def.modules['idm_net'].obs_dim),
        'action_dim': int(agent.dynamics.network.model_def.modules['idm_net'].action_dim),
        'agent': jax.device_get(flax.serialization.to_state_dict(agent)),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    temporary_path = save_path.with_suffix('.pkl.tmp')
    with temporary_path.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, save_path)
    return str(save_path)


def _read_checkpoint(path, step=0):
    from agents.pathbridger import CORE_REVISION
    path, step = resolve_checkpoint(path, step)
    with path.open('rb') as file:
        payload = pickle.load(file)
    if payload.get('format') != CHECKPOINT_FORMAT or payload.get('core_revision') != CORE_REVISION:
        raise ValueError('Expected a preserved-3999f37 checkpoint. Earlier unified checkpoints lack '
                         'the original critic/Adam state and cannot resume this computation path exactly.')
    for name in ('dynamics', 'critic'):
        if int(payload['agent'][name]['network']['step']) != step + 1:
            raise ValueError('Checkpoint filename and optimizer steps disagree.')
    return payload


def restore_agent(agent, restore_path, step=0):
    import json
    payload = _read_checkpoint(restore_path, step)
    if json.dumps(payload['config'], sort_keys=True) != json.dumps(dict(agent.config), sort_keys=True):
        raise ValueError('Resume configuration differs from the checkpoint configuration.')
    restored = flax.serialization.from_state_dict(agent, payload['agent'])
    np.random.set_state(payload['numpy_random_state'])
    random.setstate(payload['python_random_state'])
    return restored


def load_agent(path, step=0):
    """Construct from checkpoint metadata; evaluation never loads the training dataset."""
    from agents import PathBridgerAgent
    payload = _read_checkpoint(path, step)
    agent = PathBridgerAgent.create(0, np.zeros((1, payload['obs_dim']), np.float32),
                                  np.zeros((1, payload['action_dim']), np.float32), payload['config'])
    return flax.serialization.from_state_dict(agent, payload['agent'])
