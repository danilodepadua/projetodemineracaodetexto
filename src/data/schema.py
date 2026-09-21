from __future__ import annotations

TARGET_COLUMNS = [
    "formal_register",
    "thematic_coherence",
    "narrative_rhetorical_structure",
    "cohesion",
]
SPLITS = ("train", "valid", "test")

__all__ = ["TARGET_COLUMNS", "SPLITS"]
