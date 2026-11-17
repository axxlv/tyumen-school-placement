"""Ordered pre-search validation followed by a limited unprofiled timing run."""
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
for script in ['validate_optimization.py','benchmark_optimization.py']:
    subprocess.run([sys.executable,'-X','utf8',str(root/script)],cwd=root,check=True)
