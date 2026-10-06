"""Fixed training-future marginal bank, independent of the training RNG stream."""
import pickle
from pathlib import Path
import numpy as np
from utils.goal_representation import goal_representation
from utils.flax_utils import resolve_checkpoint


def make_reference_goal_bank(training_sampler, seed, size=512):
    """Use exactly the training anchor and geometric-positive marginal.

    Invoke before starting sampler threads. Saving/restoring global NumPy state
    preserves the original sampler implementation and the training batch stream.
    The caller must pass the training split, never validation or rollout data.
    """
    if int(size) < 1:
        raise ValueError('Reference bank size must be positive')
    state = np.random.get_state()
    try:
        np.random.seed(np.random.SeedSequence([int(seed), 741923]).generate_state(1)[0])
        goals = training_sampler.sample(int(size))['value_goals']
        projected = goal_representation(goals, 'phi', env_name=training_sampler.config['env_name'])
        return np.asarray(projected, dtype=np.float32)
    finally:
        np.random.set_state(state)


def checkpoint_reference_bank(path, step=0):
    """Read a trusted local CPB checkpoint's bank before constructing its template."""
    checkpoint, _ = resolve_checkpoint(path, step)
    with Path(checkpoint).open('rb') as file:
        payload = pickle.load(file)
    try:
        return np.asarray(payload['agent']['reference_goal_bank'], dtype=np.float32)
    except KeyError as exc:
        raise ValueError('Checkpoint predates calibrated CPB; raw runs cannot be resumed as calibrated runs') from exc
