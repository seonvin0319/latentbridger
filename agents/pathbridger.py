"""Deployment facade: retain the original module trees and separate JIT updates."""
from __future__ import annotations
import flax
import jax
import ml_collections
from pathbridger_core.agents.dynamics import DynamicsAgent
from pathbridger_core.agents.critic import CriticAgent
from pathbridger_core.utils.flax_utils import nonpytree_field
from utils.core_config import legacy_configs
from utils.critic_value_params import extract_critic_value_params

CORE_REVISION = '3999f3759fe6bfce1e62966bb38f0ce277add843'

class PathBridgerAgent(flax.struct.PyTreeNode):
    dynamics: DynamicsAgent
    critic: CriticAgent
    config: object = nonpytree_field()

    @classmethod
    def create(cls, seed, observations, actions, config, *, critic_batch=None):
        dc, cc = legacy_configs(config, observations.shape[-1], actions.shape[-1])
        d = DynamicsAgent.create(seed, observations, dc, ex_actions=actions)
        if critic_batch is None:
            import numpy as np
            critic_batch = {'observations': observations,
                            'action_chunk_actions': np.tile(actions, (1, 5))}
        c = CriticAgent.create(seed, critic_batch['observations'], None,
                              critic_batch['action_chunk_actions'], cc,
                              ex_goals=critic_batch.get('value_goals'))
        return cls(dynamics=d, critic=c, config=flax.core.FrozenDict(dict(config)))

    def update(self, batch):
        # Do not wrap/merge these calls in another jit. Dynamics borrows the
        # PRE-update critic value selected by the original main.py helper
        # (online modules_value unless critic._is_trl()); each original update
        # owns its Adam and EMA.
        db, cb = batch
        d, di = self.dynamics.update(
            db, critic_value_params=extract_critic_value_params(self.critic))
        c, ci = self.critic.update(cb)
        info = {**{'dynamics/' + k: v for k, v in di.items()},
                **{'critic/' + k: v for k, v in ci.items()}}
        return self.replace(dynamics=d, critic=c), info

    def sample_action_chunks(self, observations, goals, seed, *, num_candidates, temperature):
        from utils.evaluation import EvalDynamics
        policy = EvalDynamics(self.dynamics, num_candidates, temperature)
        z = policy.infer_subgoal_for_eval(observations, goals, self.critic, seed)
        trajectory = policy.plan(observations, z)['trajectory']
        return policy._idm_actions_from_trajectories(trajectory, 5)


def get_config() -> ml_collections.ConfigDict:
    """Return the paper's PBF AntMaze-medium settings as clean defaults."""

    return ml_collections.ConfigDict(
        dict(
            env_name='antmaze-medium-navigate-v0',
            endpoint_distribution='flow',
            horizon=25,
            discount=0.99,
            dynamics_p=(0.0, 0.0, 1.0, 0.0),
            critic_p=(0.0, 1.0, 0.0, 0.0),
            endpoint_value_scale=10.0,
            value_distance_weight_power=0.0,
            eval_num_candidates=8,
            eval_temperature=0.25,
        )
    )
