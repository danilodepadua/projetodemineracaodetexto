from ._common import Representation
from .catalog import (
    REPRESENTATION_CATALOG,
    REPRESENTATION_NAMES,
    RepresentationSpec,
    canonicalize_representation,
)


def build_representation(*args, **kwargs):
    """Build a representation with optional heavy dependencies loaded lazily."""
    from .traditional import build_representation as builder

    return builder(*args, **kwargs)


__all__ = [
    "REPRESENTATION_CATALOG",
    "REPRESENTATION_NAMES",
    "Representation",
    "RepresentationSpec",
    "build_representation",
    "canonicalize_representation",
]
