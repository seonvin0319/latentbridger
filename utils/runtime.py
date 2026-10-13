"""Reproducible training runtime defaults, applied before importing JAX."""
import os


def configure_training_runtime():
    options = os.environ.get('XLA_FLAGS', '')
    for name, value in [('xla_gpu_autotune_level', '0'), ('xla_gpu_deterministic_ops', 'true')]:
        if '--' + name not in options:
            options += f' --{name}={value}'
    os.environ['XLA_FLAGS'] = options.strip()
