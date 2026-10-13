"""Training runtime defaults, applied before importing JAX.

Deterministic XLA (autotune_level=0 + deterministic_ops) is opt-in via
PB_XLA_DETERMINISTIC=1. Default is off for throughput. Equivalence /
benchmark harnesses that need bit-stable GPU ops should set the env var
or pass XLA_FLAGS explicitly.
"""
import os


def configure_training_runtime():
    if os.environ.get('PB_XLA_DETERMINISTIC', '').lower() not in ('1', 'true', 'yes'):
        return
    options = os.environ.get('XLA_FLAGS', '')
    for name, value in [('xla_gpu_autotune_level', '0'), ('xla_gpu_deterministic_ops', 'true')]:
        if '--' + name not in options:
            options += f' --{name}={value}'
    os.environ['XLA_FLAGS'] = options.strip()
