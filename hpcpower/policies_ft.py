"""Regole per FTEnv (Stadio A: tipo dichiarato). Famiglia dichiarata a priori: ordine dei job (fifo | sjf | lpt) x preferenza di sezione (compute -> veloce poi efficiente; memory -> efficiente poi veloce)
x regola del livello L (sempre 1.0 | sempre 0.7 | 0.7 se D <= theta * P_max, theta in {0.7, 0.8, 0.9}). Con allow_wait=True esiste anche la variante 'W': i compute aspettano la sezione veloce."""
from .env_ft import PMAX

KEYS = {"fifo": lambda j: (j.arr, j.id), "sjf": lambda j: (j.nom, j.id), "lpt": lambda j: (-j.nom * j.n, j.id)}
LR = {"L1": None, "L07": 0.0, "th70": 0.70, "th80": 0.80, "th90": 0.90}


def make_rule(order, lr, wait=False):
    key = KEYS[order]; th = LR[lr]

    def pol(env):
        h = env.heads(); fr = [env.free(0), env.free(1)]
        for c in sorted(h, key=lambda c: key(h[c])):
            j = h[c]
            prefs = ([0] if wait else [0, 1]) if j.typ == 0 else [1, 0]
            for s in prefs:
                if fr[s] >= j.n: return 2 + 2 * c + s
        if env.allow_wait and not env.placements() is None and False: pass
        # nessun piazzamento scelto: avanza con il livello della regola
        if lr == "L1": return 0
        if lr == "L07": return 1
        return 1 if env.demand() <= th * PMAX else 0
    return pol


FT_POLICIES = {}
for _o in KEYS:
    for _l in LR:
        FT_POLICIES[f"ft_{_o}_{_l}"] = make_rule(_o, _l)
        FT_POLICIES[f"ftW_{_o}_{_l}"] = make_rule(_o, _l, wait=True)
