#!/bin/sh
# Install requirements.lock (every distribution pinned by sha256) into a new, empty virtualenv, import every
# module under planner/ and run the test suite there: a pin that lacks what the code uses fails here, not on the
# first request. `make check-install` runs it with the Makefile defaults.
#
#   scripts/check_clean_install.sh
#
# UV          the uv binary (default: uv on PATH)
# PYTHON      interpreter for the virtualenv (default 3.12: the lock is resolved for it)
# UV_OFFLINE=1  install from the local uv cache only
set -eu
cd "$(dirname "$0")/.."
UV="${UV:-uv}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$UV" venv --quiet --python "${PYTHON:-3.12}" "$work/venv"
"$UV" pip sync --quiet --python "$work/venv/bin/python" --require-hashes requirements.lock
"$work/venv/bin/python" - <<'EOF'
import importlib
import pathlib
import sys

# __main__ is the command line itself: importing it would run the program
names = sorted(
    ".".join(p.with_suffix("").parts).removesuffix(".__init__")
    for p in pathlib.Path("planner").rglob("*.py")
    if p.stem != "__main__"
)
for name in names:
    importlib.import_module(name)
print(f"python {sys.version.split()[0]}: imported {len(names)} modules ({', '.join(names)})")
EOF
"$work/venv/bin/python" -m pytest -q -p no:cacheprovider
