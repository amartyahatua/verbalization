"""Puts every stage folder on sys.path (the pipeline's scripts do this themselves at import time)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for stage in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"):
    sys.path.insert(0, os.path.join(ROOT, stage))
