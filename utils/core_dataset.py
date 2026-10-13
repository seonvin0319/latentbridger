"""Original shared-index sampling, including unused draws and fields."""
import numpy as np
from pathbridger_core.utils.datasets import Dataset, PathHGCDataset
from pathbridger_core.utils.critic_sequence_dataset import CriticSequenceDataset
from utils.core_config import legacy_configs


class CoreDataset:
    def __init__(self, data, config):
        dc, cc = legacy_configs(config, data['observations'].shape[-1], data['actions'].shape[-1])
        # Both native samplers can share the immutable dataset and its index
        # metadata. Their sampled batches and RNG order remain independent.
        native_data = Dataset.create(**dict(data))
        self.dynamics = PathHGCDataset(native_data, dc)
        self.critic = CriticSequenceDataset(native_data, cc)
        self.valid_starts = np.asarray(np.intersect1d(
            self.dynamics.path_valid_idxs, self.critic.valid_starts, assume_unique=False), dtype=np.int64)
        if not len(self.valid_starts):
            raise ValueError('No shared valid starts across dynamics and critic datasets.')

    def sample(self, batch_size):
        picked = np.random.randint(len(self.valid_starts), size=batch_size)
        idxs = self.valid_starts[picked]
        db = self.dynamics.sample(batch_size, idxs=idxs)
        cb = self.critic.sample(batch_size, idxs=idxs)
        return db, cb
