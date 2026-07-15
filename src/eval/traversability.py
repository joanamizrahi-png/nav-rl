"""Load per-class traversability scores from config/traversability.yaml.

The yaml lives at repo-root/config/traversability.yaml. Each entry has a
class ID, a name, a score in [0, 1], and a note explaining the choice.

Usage:
    scores = load_traversability()          # np.ndarray shape (30,) float32
    scores[grass_class_id] -> 0.95
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# Fixed size of the taxonomy. Must match sam3_precompute_labels.CLASSES and
# diffsynth.utils.semantics.CLASS_COLORS.
NUM_CLASSES = 30


def _find_config_path() -> Path:
    """Locate config/traversability.yaml relative to this file's repo root.

    We walk up from __file__ until we find a `config/` sibling of `src/`.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        candidate = parent / "config" / "traversability.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find config/traversability.yaml relative to eval/traversability.py; "
        "expected repo-root/config/traversability.yaml"
    )


def load_traversability(path: Optional[Path] = None) -> np.ndarray:
    """Return a (NUM_CLASSES,) float32 array indexed by class ID.

    Missing class IDs default to 0.0 (conservative). Warns if any are missing.
    """
    path = path or _find_config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)

    scores = np.zeros(NUM_CLASSES, dtype=np.float32)
    seen = set()
    for cid, entry in raw.items():
        cid = int(cid)
        if not (0 <= cid < NUM_CLASSES):
            print(f"[traversability] WARNING: class id {cid} out of range [0, {NUM_CLASSES}); skipping")
            continue
        scores[cid] = float(entry["score"])
        seen.add(cid)

    missing = set(range(NUM_CLASSES)) - seen
    if missing:
        print(f"[traversability] WARNING: {len(missing)} classes missing from yaml, defaulting to 0.0: {sorted(missing)}")

    return scores


def load_class_names(path: Optional[Path] = None) -> list[str]:
    """Return class names indexed by class ID (for plotting / logs)."""
    path = path or _find_config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)
    names = ["unknown"] * NUM_CLASSES
    for cid, entry in raw.items():
        cid = int(cid)
        if 0 <= cid < NUM_CLASSES:
            names[cid] = str(entry["name"])
    return names
