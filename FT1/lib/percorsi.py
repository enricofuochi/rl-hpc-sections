"""Path setup: lets the FT1 scripts run from ANY working directory.
Every script starts with:  import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
It finds the repository root (the folder that contains `hpcpower/`) and puts on sys.path: FT1/lib, FT1 (ft1lib), the root (hpcpower), muzero/ (slim_aux), FT1/muzero (mz_eval_ft1)."""
import sys
from pathlib import Path

FT1 = Path(__file__).resolve().parent.parent
LIB = FT1 / "lib"
OUTPUT = FT1 / "output"
RUNS = FT1 / "runs"
CONFIG = FT1 / "config"


def _root():
    for c in (FT1.parent, *FT1.parents):
        if (c / "hpcpower").is_dir(): return c
    raise SystemExit("Cannot find 'hpcpower/': the FT1 folder must sit inside the repository, next to hpcpower/ and muzero/.")


RADICE = _root()
for _p in (FT1 / "muzero", RADICE / "muzero", RADICE, FT1, LIB):      # the last one inserted ends up first
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
OUTPUT.mkdir(exist_ok=True)
