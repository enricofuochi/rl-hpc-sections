"""Path setup: lets the FT0 scripts run from ANY working directory.
Every script starts with:  import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
It finds the repository root (the folder that contains `hpcpower/`) and puts on sys.path: FT0/lib, the root (hpcpower), muzero/ (slim_aux)."""
import sys
from pathlib import Path

FT0 = Path(__file__).resolve().parent.parent
LIB = FT0 / "lib"
OUTPUT = FT0 / "output"
CONFIG = FT0 / "config"


def _root():
    for c in (FT0.parent, *FT0.parents):
        if (c / "hpcpower").is_dir(): return c
    raise SystemExit("Cannot find 'hpcpower/': the FT0 folder must sit inside the repository, next to hpcpower/ and muzero/.")


RADICE = _root()
for _p in (RADICE / "muzero", RADICE, LIB):      # the last one inserted ends up first
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
OUTPUT.mkdir(exist_ok=True)
