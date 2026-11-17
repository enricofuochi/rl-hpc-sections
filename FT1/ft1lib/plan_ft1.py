"""Pianificatore PRIVILEGIATO (come PLAN di FT0): copia l'ambiente (stato esatto), prova le azioni legali, poi continua con la regola. depth=1: una azione di lookahead; depth=2: due.
Non e' un'agente onesto (conosce le variabili nascoste): serve da riferimento del margine raggiungibile. L'ambiente e' deterministico, quindi il rollout e' esatto."""
import copy
import numpy as np


def _resto(e, regola):
    while True:
        _, _, te, tr, _ = e.step(int(regola(e)))
        if te or tr: return e.ret


def _valore(e, regola, depth):
    """Costo totale (ore) dell'episodio se da e si fanno `depth` passi ottimi e poi si segue la regola."""
    if depth == 0: return _resto(e, regola)
    best = 1e18
    for a in np.flatnonzero(e.action_masks()):
        e2 = copy.deepcopy(e); _, _, te, tr, _ = e2.step(int(a))
        c = e2.ret if (te or tr) else _valore(e2, regola, depth - 1)
        best = min(best, c)
    return best


def azione_plan(e, regola, depth=1):
    best, bc = None, 1e18
    for a in np.flatnonzero(e.action_masks()):
        e2 = copy.deepcopy(e); _, _, te, tr, _ = e2.step(int(a))
        c = e2.ret if (te or tr) else _valore(e2, regola, depth - 1)
        if c < bc - 1e-12: best, bc = int(a), c
    return best
