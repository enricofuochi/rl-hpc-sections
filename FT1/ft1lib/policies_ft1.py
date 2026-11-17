"""Regole per FT1. Piazzamento come in FT0 (ordine fifo|sjf|lpt; compute -> veloce poi efficiente, memory -> efficiente poi veloce; senza attesa).
Livello: 'L1' = resta a 1.0 | 'trkXX' = TRACKING: bersaglio L* = D/P_max + XX/100 (D = domanda di potenza dei job in esecuzione); si fa la mossa legale che porta L piu' vicino a L*.
Con la rampa il tracking insegue la domanda con al massimo delta per tick."""
from hpcpower.env_ft import PMAX

KEYS = {"fifo": lambda j: (j.arr, j.id), "sjf": lambda j: (j.nom, j.id), "lpt": lambda j: (-j.nom * j.n, j.id)}
LR = {"L1": None, "trk00": 0.0, "trk05": 0.05, "trk10": 0.10, "trk20": 0.20}


def piazza(env, key):
    """Azione di piazzamento della regola, o None se nessun piazzamento e' legale."""
    h = env.heads(); fr = [env.free(0), env.free(1)]
    for c in sorted(h, key=lambda c: key(h[c])):
        j = h[c]
        for s in ([0, 1] if j.typ == 0 else [1, 0]):
            if fr[s] >= j.n: return env.na + 2 * c + s
    return None


def mossa_verso(env, bersaglio):
    """Mossa di avanzamento legale che porta L piu' vicino al bersaglio (a parita' la prima: scendere)."""
    mk = env.action_masks(); b = min(1.0, max(env.lmin, bersaglio)); best, bd = None, 1e9
    for i in range(env.na):
        if not mk[i]: continue
        d = abs(env.new_L(i) - b)
        if d < bd - 1e-12: best, bd = i, d
    return best


def make_rule(order, lr):
    key = KEYS[order]; m = LR[lr]

    def pol(env):
        p = piazza(env, key)
        if p is not None: return p
        return mossa_verso(env, 1.0 if m is None else env.demand() / PMAX + m)
    return pol


def regola_th(soglia=0.8, basso=0.7):
    """Regola di soglia come FT0 (sjf; L = basso se D <= soglia*P_max, altrimenti 1.0), da usare con delta = 1 - basso e lmin = basso: riproduce FT0 (serve solo ai test)."""
    key = KEYS["sjf"]

    def pol(env):
        p = piazza(env, key)
        if p is not None: return p
        return mossa_verso(env, basso if env.demand() <= soglia * PMAX else 1.0)
    return pol


FT1_POLICIES = {f"ft1_{o}_{l}": make_rule(o, l) for o in KEYS for l in LR}
