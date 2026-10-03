"""Every module imports on its own, first, in a fresh interpreter: an import cycle that only
works in one order (the order the test suite happens to use) fails here."""
from __future__ import annotations

import pkgutil
import subprocess
import sys

import gpu_broker

MODULES = sorted(m.name for m in pkgutil.walk_packages(gpu_broker.__path__, "gpu_broker.")
                 if not m.name.endswith("__main__"))
PROBE = """
import importlib, sys
failed = []
for name in sys.argv[1:]:
    for loaded in [m for m in sys.modules if m == "gpu_broker" or m.startswith("gpu_broker.")]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except Exception as e:
        failed.append(f"{name}: {e!r}")
print("\\n".join(failed))
"""


def test_every_module_imports_first():
    assert "gpu_broker.settings" in MODULES and "gpu_broker.gpu.auto" in MODULES
    out = subprocess.run([sys.executable, "-c", PROBE, *MODULES], capture_output=True, text=True,
                         timeout=120, check=True).stdout
    assert out.strip() == ""
