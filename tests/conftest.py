"""
Pytest configuration: add project root to sys.path so app and src can be imported.

Run tests from project root: pytest tests/ -v
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
