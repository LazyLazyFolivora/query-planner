"""Load by file path; never imports a host application package named ``src``.

This module contains no network/model client. Default decisions need only NumPy.
"""
from __future__ import annotations
import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent / "src"
_VERSION = hashlib.sha256(Path(__file__).read_bytes() + b"".join(
    str(path.relative_to(_ROOT)).encode() + b"\0" + path.read_bytes()
    for path in sorted(_ROOT.rglob("*.py"))
)).hexdigest()
_PACKAGE = "_query_planner_" + _VERSION
if _PACKAGE not in sys.modules:
    package = ModuleType(_PACKAGE)
    package.__path__ = [str(_ROOT)]
    package.__package__ = _PACKAGE
    sys.modules[_PACKAGE] = package
_api = importlib.import_module(_PACKAGE + ".integration.external")
SCHEMA_VERSION = _api.SCHEMA_VERSION
POLICY_VERSION = _VERSION

def decide(request: dict) -> dict:
    """Choose one supplied action or stop; the host executes and persists it."""
    result = _api.decide(request)
    result["policy"]["version"] = POLICY_VERSION
    return result

def create_policy(*, selector="heuristic", stop="rule"):
    """Explicit opt-in to existing local model heads; never downloads encoders."""
    return _api.HostPolicy(selector=selector, stop=stop, version=POLICY_VERSION)

def load_public_questions(path):
    """Read only public task-construction-v2 rows, without oracle-derived hints."""
    module = importlib.import_module(_PACKAGE + ".integration.public_questions")
    return module.load_public_questions(path)
