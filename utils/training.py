"""Training-only storage reuse around the unchanged, separate native updates.

The public agent.update remains non-destructive. These helpers are for loops
that relinquish the previous state on every update. Never donate state that
another caller still needs for evaluation, comparison, or saving.
"""
import jax

from pathbridger_core.agents.dynamics import DynamicsAgent
from pathbridger_core.agents.critic import CriticAgent
from utils.critic_value_params import extract_critic_value_params

# Compile exactly the native function bodies at their original JIT boundaries.
# Only ownership of the input state buffers differs; no outer JIT or new math.
_dynamics_update = jax.jit(DynamicsAgent.update.__wrapped__, donate_argnums=(0,))
_critic_update = jax.jit(CriticAgent._update_impl.__wrapped__, donate_argnums=(0,))


def own_training_state(agent):
    """Detach initial/restored aliases once before consuming state buffers.

    Native initialization can share Adam zero and target buffers between leaves.
    A donated argument cannot contain the same buffer twice. Copy each device
    leaf without doing arithmetic or consuming RNG. Host leaves are transferred
    separately by the native JIT when first used.
    """
    return jax.tree_util.tree_map(
        lambda value: value.copy() if isinstance(value, jax.Array) else value, agent)


def update_training_state(agent, batch):
    """Return updated state/metrics, consuming the previous agent's buffers."""
    db, cb = batch
    # Retain the public critic's validation before entering its native JIT.
    # Validate before donating dynamics so an invalid batch leaves state intact.
    agent.critic._validate_trl_batch(cb)
    # Same pre-update critic selection as legacy main.py / PathBridgerAgent.update.
    d, di = _dynamics_update(
        agent.dynamics, db,
        critic_value_params=extract_critic_value_params(agent.critic))
    c, ci = _critic_update(agent.critic, cb)
    info = {**{'dynamics/' + key: value for key, value in di.items()},
            **{'critic/' + key: value for key, value in ci.items()}}
    return agent.replace(dynamics=d, critic=c), info
