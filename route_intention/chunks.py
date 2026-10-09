"""Long-horizon route windows. The encoder never sees the goal, reward, or absolute state."""

from __future__ import annotations

import numpy as np

from intention_pb.pb_io import chunk_valid_starts

from route_intention.common import H_ROUTE

# Multi-step relative endpoints. Only horizons that fit inside H_route are used.
_BASE_HORIZONS = (5, 10, 20)


def prediction_horizons(horizon: int = H_ROUTE) -> tuple[int, ...]:
    hs = [h for h in _BASE_HORIZONS if 0 < h <= int(horizon)]
    if int(horizon) not in hs:
        hs.append(int(horizon))
    return tuple(hs)


def waypoint_offsets(horizon: int = H_ROUTE) -> tuple[int, ...]:
    h = int(horizon)
    offs = []
    for num, den in ((1, 4), (1, 2), (3, 4)):
        t = (num * h) // den
        if 0 < t < h and t not in offs:
            offs.append(t)
    return tuple(offs)


def valid_route_starts(terminals: np.ndarray, horizon: int = H_ROUTE) -> np.ndarray:
    """Starts ``t`` whose window ``[t, t+horizon]`` stays inside one episode."""
    if int(horizon) < 1:
        raise ValueError(f'horizon must be positive, got {horizon}')
    return chunk_valid_starts(np.asarray(terminals), int(horizon))


def window_views(observations: np.ndarray, actions: np.ndarray, starts: np.ndarray, horizon: int = H_ROUTE):
    """Return ``(obs_win [B,H+1,D], act_win [B,H,A])`` without copying more than the gather."""
    starts = np.asarray(starts, dtype=np.int64)
    horizon = int(horizon)
    obs_off = starts[:, None] + np.arange(horizon + 1)[None, :]
    act_off = starts[:, None] + np.arange(horizon)[None, :]
    obs = np.asarray(observations, dtype=np.float32)[obs_off]
    act = np.asarray(actions, dtype=np.float32)[act_off]
    return obs, act


def encoder_sequence(obs_win: np.ndarray, act_win: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step features ``[a_{t+j}, s_{t+j+1} - s_t]``, both z-scored. No absolute state, no goal.

    ``obs_win`` is ``s_t .. s_{t+H}`` and ``act_win`` is ``a_t .. a_{t+H-1}``.
    """
    obs = np.asarray(obs_win, dtype=np.float32)
    act = np.asarray(act_win, dtype=np.float32)
    rel = obs[:, 1:, :] - obs[:, :1, :]
    a_n = (act - stats['action_mean']) / stats['action_std']
    r_n = (rel - stats['rel_mean']) / stats['rel_std']
    return np.concatenate([a_n, r_n], axis=-1).astype(np.float32)


def state_t_normalized(obs_win: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Decoder-only absolute state. Not an encoder input."""
    s = np.asarray(obs_win, dtype=np.float32)[:, 0, :]
    return ((s - stats['state_mean']) / stats['state_std']).astype(np.float32)


def target_matrix(obs_win: np.ndarray, act_win: np.ndarray, stats: dict[str, np.ndarray], horizon: int = H_ROUTE) -> np.ndarray:
    """Z-scored multi-horizon endpoints, coarse waypoints, and the mean action. Shape ``[B, T]``."""
    obs = np.asarray(obs_win, dtype=np.float32)
    act = np.asarray(act_win, dtype=np.float32)
    rel = obs[:, 1:, :] - obs[:, :1, :]
    parts = []
    for h in prediction_horizons(horizon):
        parts.append((rel[:, h - 1, :] - stats['endpoint_mean']) / stats['endpoint_std'])
    for t in waypoint_offsets(horizon):
        parts.append((rel[:, t - 1, :] - stats['endpoint_mean']) / stats['endpoint_std'])
    mean_a = act.mean(axis=1)
    parts.append((mean_a - stats['action_mean']) / stats['action_std'])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def target_dim(horizon: int, state_dim: int, action_dim: int) -> int:
    n_vec = len(prediction_horizons(horizon)) + len(waypoint_offsets(horizon))
    return n_vec * int(state_dim) + int(action_dim)


def compute_route_stats(observations: np.ndarray, actions: np.ndarray, starts: np.ndarray, horizon: int = H_ROUTE) -> dict[str, np.ndarray]:
    obs_win, act_win = window_views(observations, actions, starts, horizon)
    rel = obs_win[:, 1:, :] - obs_win[:, :1, :]
    end = rel[:, -1, :]
    act = act_win.reshape(-1, act_win.shape[-1])
    floor = np.float32(1e-3)

    def _ms(x):
        return x.mean(0).astype(np.float32), np.maximum(x.std(0), floor).astype(np.float32)

    a_mean, a_std = _ms(act)
    r_mean, r_std = _ms(rel.reshape(-1, rel.shape[-1]))
    e_mean, e_std = _ms(end)
    s_mean, s_std = _ms(obs_win[:, 0, :])
    return dict(
        action_mean=a_mean, action_std=a_std,
        rel_mean=r_mean, rel_std=r_std,
        endpoint_mean=e_mean, endpoint_std=e_std,
        state_mean=s_mean, state_std=s_std,
    )
