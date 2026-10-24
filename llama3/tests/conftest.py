"""Puts every stage folder on sys.path (the pipeline's scripts do this themselves at import time)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for stage in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"):
    sys.path.insert(0, os.path.join(ROOT, stage))


# The real `jlens` package (pip install git+https://github.com/anthropics/jacobian-lens) is only needed to BUILD
# directions; tests never do, so stub it when absent so the test-suite runs on machines without it.
import importlib.util, sys, types
if importlib.util.find_spec("jlens") is None:
    sys.modules["jlens"] = types.ModuleType("jlens")
