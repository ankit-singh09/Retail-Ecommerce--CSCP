"""Expose RFM feature engineering functions from sibling module `src/features.py`."""

import importlib.util
from pathlib import Path

_features_file = Path(__file__).resolve().parents[1] / "features.py"
_spec = importlib.util.spec_from_file_location("src._features_module", _features_file)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_module)

build_rfm_features = _module.build_rfm_features
build_features = _module.build_features
save_rfm_features = _module.save_rfm_features
main = _module.main

__all__ = ["build_rfm_features", "build_features", "save_rfm_features", "main"]
