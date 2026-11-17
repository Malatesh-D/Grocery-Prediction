"""
Convenience entrypoint script to execute runtime tests and generate all terminal screenshots.
Usage:
    python run_runtime_suite.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture_runtime_screenshots import generate_all_runtime_screenshots

if __name__ == "__main__":
    generate_all_runtime_screenshots()
