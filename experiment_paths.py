"""Locate shared project data beside the visual-memory folder."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CS230_PROJECT_ROOT', HERE.parent)).expanduser().resolve()
