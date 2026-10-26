"""Execute a module from this frozen code tree, even when cwd has live code."""
import os
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python /absolute/frozen/code/multinav/bootstrap.py MODULE [ARGS]')
    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    os.environ['PYTHONPATH'] = str(code_root) + os.pathsep + os.environ.get('PYTHONPATH', '')
    module = sys.argv.pop(1)
    sys.argv[0] = module
    runpy.run_module(module, run_name='__main__', alter_sys=True)
