"""Verify the built wheel away from the source tree."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wheel', type=Path)
    arg = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(arg.wheel) as wheel:
            names = wheel.namelist()
            assert 'pathbridger_core/PROVENANCE.json' in names
            assert sum(n.startswith('configs/pb') and n.endswith('.py') and not n.endswith('__init__.py') for n in names) == 16
            wheel.extractall(temp)
        code = '''
import json,hashlib,importlib
from pathlib import Path
import pathbridger_core,main
root=Path(pathbridger_core.__file__).parent
manifest=json.loads((root/'PROVENANCE.json').read_text())
for name,entry in manifest['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==entry['vendored_sha256'],name
assert Path(main.__file__).parent==root.parent
from agents import PathBridgerAgent
assert Path(importlib.import_module('agents.pathbridger').__file__).parent.parent==root.parent
from utils.core_config import legacy_configs
for family in ['pbf','pbg']:
 for name in ['antmaze_medium','antmaze_large','cube_single','cube_double','cube_triple','puzzle_3x3','puzzle_4x4','scene']:
  config=importlib.import_module(f'configs.{family}.{name}').get_config()
  assert config.horizon>=5
print('INSTALLED_PACKAGE_PASS')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=temp, text=True, capture_output=True, check=True)
        print(result.stdout)
    Path('validation/package.json').write_text(json.dumps({'passed': True, 'wheel': str(arg.wheel),
        'isolated_imports': True, 'packaged_core_hashes_match': True, 'configs': 16}, indent=2)+'\n')


if __name__ == '__main__':
    main()
