"""Record the precise code/runtime/config used by a deployment run."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from agents.pathbridger import CORE_REVISION


def write_provenance(run_dir, agent):
    import jax
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ['pathbridger_core', 'agents', 'utils', 'configs', 'envs']
             for p in sorted((root / folder).rglob('*.py'))}
    for name in ['main.py', 'evaluate.py']:
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    record = dict(core_revision=CORE_REVISION, deployment_revision=revision, sources=files,
                  runtime={x: importlib.metadata.version(x) for x in ['jax', 'jaxlib', 'flax', 'optax', 'numpy', 'ogbench']},
                  backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                  xla_flags=os.environ.get('XLA_FLAGS', ''), dynamics=dict(agent.dynamics.config),
                  critic=dict(agent.critic.config), actor_spi=False,
                  core_compatibility_patches=json.loads((root / 'pathbridger_core/PROVENANCE.json').read_text()).get('compatibility_patches', []),
                  evaluation_protocol='3999-idm-strict-reset-v2')
    (Path(run_dir) / 'provenance.json').write_text(json.dumps(record, indent=2, default=list)+'\n')
