"""Install a wheel in a fresh venv, then run runtime and optional reference checks."""

import glob
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


python = Path('.wheel-test') / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if '--reference' in sys.argv:
    run(python, '-m', 'pip', 'install', '-r', 'benchmarks/reference-requirements.txt')
    run(python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py')
    with tempfile.TemporaryDirectory() as output:
        run(python.resolve(), Path('examples/plot_comparison.py').resolve(),
            cwd=output, env={**os.environ, 'MPLBACKEND': 'Agg'})
else:
    venv.create(python.parent.parent, with_pip=True)
    wheels = glob.glob('dist/*.whl')
    assert len(wheels) == 1, wheels
    run(python, '-m', 'pip', 'install', wheels[0])
    run(python, '-c', 'import importlib.util; assert all(importlib.util.find_spec(x) is None '
        'for x in ("rliable", "arch", "scipy", "pandas"))')
    run(python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_runtime.py')
