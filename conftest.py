"""Repo-root pytest configuration.

WHY THIS EXISTS. There was no conftest.py, no pytest.ini and no
[tool.pytest.ini_options] anywhere in XPU-RT, which had two consequences worth
fixing rather than documenting:

1. `xpu-rt/` is not importable as a package -- every script prepends it to
   sys.path itself -- so `pytest` from the repo root could only collect the
   tests that happened to do that themselves. Doing it once here makes
   `pytest` at the root mean what a reader expects.
2. MOSEK's licence lives in the repo but nothing pointed at it, so two tests
   failed with "MOSEK is not installed" on a checkout that has it. Set it when
   the file is there and the environment has not already chosen one.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "xpu-rt"), os.path.join(_ROOT, "scripts"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_lic = os.path.join(_ROOT, "mosek.lic")
if os.path.exists(_lic) and not os.environ.get("MOSEKLM_LICENSE_FILE"):
    os.environ["MOSEKLM_LICENSE_FILE"] = _lic

# CP-SAT: pin the worker count for tests so a run is reproducible on any host.
# The docs mandate XPURT_CPSAT_WORKERS=0 (auto) to beat greedy, and
# scheduler_cpsat says in as many words that >1 worker under a time limit is
# NOT deterministic. A test suite wants the deterministic setting; a benchmark
# wants the fast one, and it can still ask for it explicitly.
os.environ.setdefault("XPURT_CPSAT_WORKERS", "1")
