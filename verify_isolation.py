from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
FORBIDDEN_TOP_LEVELS = {
    "stfm"
}


def check_source_imports() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_TOP_LEVELS:
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {name}")
    if violations:
        raise RuntimeError("imports escape the package:\n" + "\n".join(violations))


def check_module_origins() -> None:
    modules = (
        "data.loader",
        "model.common",
        "model.neural_field",
        "model.diffusion",
        "stage1_grounded",
        "training",
        "experiment",
    )
    package = PACKAGE_ROOT.name
    for suffix in modules:
        name = f"{package}.{suffix}"
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve module: {name}")
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(PACKAGE_ROOT):
            raise RuntimeError(f"{name} resolved outside package: {origin}")


if __name__ == "__main__":
    check_source_imports()
    check_module_origins()
    print(f"isolation verified: {PACKAGE_ROOT}")
