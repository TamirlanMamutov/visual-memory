"""Locate shared project data beside the visual-memory folder."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CS230_PROJECT_ROOT', HERE.parent)).expanduser().resolve()


def project_dir(name):
    """Generated folders (GPU environment, caches) live in the project folder: ROOT/<name>.
    Copies made before this convention sat at the drive root (<drive>/<name>); that
    location is used only while it exists and ROOT/<name> does not."""
    new, old = ROOT / name, Path(ROOT.anchor) / name
    return old if old.is_dir() and not new.exists() else new
