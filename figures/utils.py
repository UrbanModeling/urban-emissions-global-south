"""
utils.py
Shared helpers for vis scripts.
"""

from pathlib import Path


def next_path(base: str) -> str:
    """Return a non-conflicting file path.

    If base does not exist, return it unchanged.
    Otherwise append _v2, _v3, ... until a free slot is found.

    Example:
        next_path('fig/growth_scatter.png')
        -> 'fig/growth_scatter.png'          # if not exists
        -> 'fig/growth_scatter_v2.png'       # if v1 exists
        -> 'fig/growth_scatter_v3.png'       # if v1 and v2 exist
    """
    p = Path(base)
    if not p.exists():
        return base
    stem, suffix, parent = p.stem, p.suffix, p.parent
    v = 2
    while True:
        candidate = parent / f'{stem}_v{v}{suffix}'
        if not candidate.exists():
            return str(candidate)
        v += 1
