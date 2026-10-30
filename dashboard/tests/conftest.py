"""Pytest configuration for the dashboard package.

``dashboard/lib`` is a package whose modules use relative imports
(``from . import metrics``). ``generate.py`` makes it importable by putting the
``dashboard`` directory on ``sys.path`` and importing ``from lib import ...``.
Tests replicate that here so ``from lib import portfolio`` resolves.
"""
import os
import sys

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)
