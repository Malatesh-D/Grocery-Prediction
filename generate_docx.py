"""
Convenience entrypoint script to compile the complete Word (.docx) Academic Report.
Usage:
    python generate_docx.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_docx_report import generate_word_report

if __name__ == "__main__":
    generate_word_report()
