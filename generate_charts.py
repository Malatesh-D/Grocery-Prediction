"""
Convenience entrypoint script to run the complete visualization generator suite.
Usage:
    python generate_charts.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_visualizations import run_all_visualizations

if __name__ == "__main__":
    run_all_visualizations()
