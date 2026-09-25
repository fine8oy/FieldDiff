from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_main():
    if __package__:
        from .experiment import main

        return main

    # Support ``python path/to/run.py`` without resolving any implementation
    # modules from the parent FieldDiff checkout.
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    module = importlib.import_module(f"{package_dir.name}.experiment")
    return module.main


if __name__ == "__main__":
    _load_main()()
