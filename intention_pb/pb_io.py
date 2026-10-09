"""Load original PathBridger runs (flags.json + params_<step>.pkl), read-only."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from intention_pb import ensure_pb_code_path

ensure_pb_code_path()

from agents.dynamics import DynamicsAgent
from eval_checkpoint import _build_configs
from main import _create_critic_agent, _intersect_valid_starts, _make_critic_dataset, _sample_shared_idxs
from utils.datasets import Dataset, PathHGCDataset
from utils.goal_representation import infer_phi_goal_obs_indices, normalize_phi_goal_obs_indices
from utils.run_io import load_checkpoint_pkl

from intention_pb.common import FINAL_STEP, pb_checkpoint_paths


@dataclasses.dataclass
class PBBundle:
    run_dir: Path
    step: int
    env_name: str
    env: Any
    train: dict | None
    val: dict
    dynamics: DynamicsAgent
    critic: Any
    dynamics_config: Any
    critic_config: Any
    flags: dict

    @property
    def horizon(self) -> int:
        return int(self.dynamics_config['dynamics_N'])

    @property
    def idm_horizon(self) -> int:
        return int(self.critic_config['action_chunk_horizon'])

    def critic_value_params(self):
        """Exactly what PB's main loop passes to ``DynamicsAgent.update`` (TRL target value)."""
        params = self.critic.network.params
        if hasattr(self.critic, '_is_trl') and bool(self.critic._is_trl()):
            return params['modules_target_value']
        return params['modules_value']


def _ogbench_data_dir(dataset_dir: str) -> str:
    return os.path.expanduser(dataset_dir) if dataset_dir else os.path.expanduser('~/.ogbench/data')


def load_plain_datasets(env_name: str, dataset_dir: str = '', *, train: bool = True):
    import ogbench

    data_dir = _ogbench_data_dir(dataset_dir)
    val_path = os.path.join(data_dir, f'{env_name}-val.npz')
    train_path = os.path.join(data_dir, f'{env_name}.npz')
    for p in ([train_path] if train else []) + [val_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    val = dict(ogbench.load_dataset(val_path, compact_dataset=True))
    tr = dict(ogbench.load_dataset(train_path, compact_dataset=True)) if train else None
    return tr, val


def make_env(env_name: str):
    import ogbench

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env.reset()
    return env


def load_pb(run_dir: Path, step: int = FINAL_STEP, *, need_train: bool = True, need_env: bool = True) -> PBBundle:
    run_dir = Path(run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(flags_path)
    ckpts = pb_checkpoint_paths(run_dir, step)
    for p in ckpts.values():
        if not p.is_file():
            raise FileNotFoundError(f'PB checkpoint missing: {p}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    fg = root['flags']
    seed = int(fg['seed'])
    dynamics_config, critic_config, _actor_config = _build_configs(root, fg)
    if critic_config.get('frame_stack') is not None:
        raise ValueError('frame_stack is not supported by the intention experiment.')
    env_name = str(fg['env_name'])
    train, val = load_plain_datasets(env_name, str(fg.get('dataset_dir', '')), train=need_train)
    env = make_env(env_name) if need_env else None
    ref = val
    obs_dim = int(np.asarray(ref['observations']).shape[-1])
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    if not phi_idxs:
        phi_idxs = infer_phi_goal_obs_indices(env_name, obs_dim)
        critic_config['phi_goal_obs_indices'] = phi_idxs
        dynamics_config['phi_goal_obs_indices'] = phi_idxs
    action_dim = int(np.asarray(ref['actions']).shape[-1])
    critic_config['action_dim'] = action_dim

    # Example batches are only used for shapes; the small validation split suffices.
    dyn_ds = PathHGCDataset(Dataset.create(**val), dynamics_config)
    crit_ds = _make_critic_dataset(val, critic_config)
    common = _intersect_valid_starts(dyn_ds, crit_ds)
    rs = np.random.get_state()
    np.random.seed(seed)
    ex_idxs = _sample_shared_idxs(common, int(dynamics_config['batch_size']))
    ex_dyn = dyn_ds.sample(len(ex_idxs), idxs=ex_idxs)
    ex_crit = crit_ds.sample(len(ex_idxs), idxs=ex_idxs)
    np.random.set_state(rs)
    dynamics = DynamicsAgent.create(
        seed,
        jnp.asarray(ex_dyn['observations'], dtype=jnp.float32),
        dynamics_config,
        ex_actions=jnp.asarray(ex_dyn['actions'], dtype=jnp.float32),
    )
    critic = _create_critic_agent(seed, ex_crit, critic_config)
    dynamics = load_checkpoint_pkl(dynamics, ckpts['dynamics'])
    critic = load_checkpoint_pkl(critic, ckpts['critic'])
    return PBBundle(
        run_dir=run_dir,
        step=int(step),
        env_name=env_name,
        env=env,
        train=train,
        val=val,
        dynamics=dynamics,
        critic=critic,
        dynamics_config=dynamics_config,
        critic_config=critic_config,
        flags=fg,
    )


def chunk_valid_starts(terminals: np.ndarray, horizon: int) -> np.ndarray:
    """Indices ``t`` with ``t + horizon`` inside the same episode (compact OGBench layout)."""
    terminals = np.asarray(terminals)
    (term_locs,) = np.nonzero(terminals > 0)
    starts = np.concatenate([[0], term_locs[:-1] + 1])
    out = []
    for s, e in zip(starts, term_locs):
        last = int(e) - int(horizon)
        if last >= int(s):
            out.append(np.arange(int(s), last + 1))
    return np.concatenate(out).astype(np.int64)
