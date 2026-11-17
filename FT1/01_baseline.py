"""Baseline di FT1 (play): regole e pianificatore privilegiato sugli stessi seed che userano gli agenti. Deterministico, nessun agente da caricare (PPO si aggiunge da solo se esiste).
  1 = calibrazione: le 15 regole (3 ordini x 5 livelli) sui seed di calibrazione 900-929  -> la migliore e' la REGOLA di riferimento (ft1lib/config_ft1.py)
  2 = baseline: regola di riferimento, PLAN 1 passo e PLAN 2 passi su validazione 1000-1029 e test 9000-9099 (+ PPO se ha runs/ppo_ft1/ppo_ft1_best_r0.zip)
  3 = tutto
Scrive output/baseline_calibrazione.csv e output/baseline_<val|test>.csv (costo in ore per seed) e stampa le tabelle con le differenze appaiate dalla regola (bootstrap IC95)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import csv, time
import numpy as np
from multiprocessing import Pool
from ft1lib.config_ft1 import KW, REGOLA, SEED_CAL, SEED_VAL, SEED_TEST
from ft1lib.factory_ft1 import make_env
from ft1lib.policies_ft1 import FT1_POLICIES
from ft1lib.plan_ft1 import azione_plan

N_PROC = 2          # processi in parallelo (aumenta se il Mac ha piu' core)


def costo(args):
    nome, seed = args; e = make_env(KW); e.reset(seed=seed); R = FT1_POLICIES[REGOLA]
    if nome in FT1_POLICIES: pol = FT1_POLICIES[nome]
    elif nome == "PLAN 1 passo": pol = lambda e: azione_plan(e, R, 1)
    elif nome == "PLAN 2 passi": pol = lambda e: azione_plan(e, R, 2)
    else: raise ValueError(nome)
    while True:
        _, _, te, tr, _ = e.step(int(pol(e)))
        if te or tr: return nome, seed, e.ret


def ppo_costi(seeds):
    p = percorsi.RUNS / "ppo_ft1" / "ppo_ft1_best_r0"
    if not pathlib.Path(str(p) + ".zip").exists(): return None
    import hpcpower.ppo as P
    ag = P.Agent.load(str(p)); out = {}
    for s in seeds:
        e = make_env(KW); e.reset(seed=s)
        while True:
            _, _, te, tr, _ = e.step(int(ag.act(e)))
            if te or tr: out[s] = e.ret; break
    return out


def boot(d, B=20000):
    rng = np.random.default_rng(20261109); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)


def tabella(res, rif, titolo):
    seeds = sorted(next(iter(res.values()))); r = np.array([res[rif][s] for s in seeds])
    print(f"\n== {titolo} | {len(seeds)} seed | riferimento: {rif} ({r.mean():.4f} h)"); print(f"{'politica':26s} {'costo (h)':>10s} {'vs rif.':>9s}   IC95 (appaiato)")
    for n, d in sorted(res.items(), key=lambda x: np.mean(list(x[1].values()))):
        a = np.array([d[s] for s in seeds]); x = (a - r); lo, hi = boot(x); print(f"{n:26s} {a.mean():10.4f} {100 * x.mean() / r.mean():+8.1f}%   [{100 * lo / r.mean():+.1f}, {100 * hi / r.mean():+.1f}]")


def scrivi(res, f):
    seeds = sorted(next(iter(res.values())))
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["seed"] + list(res)); [w.writerow([s] + [f"{res[n][s]:.6f}" for n in res]) for s in seeds]


def esegui(nomi, seeds, pool):
    res = {n: {} for n in nomi}
    for n, s, c in pool.imap_unordered(costo, [(n, s) for n in nomi for s in seeds], chunksize=2): res[n][s] = c
    return res


if __name__ == "__main__":
    print(__doc__.split("\n")[0]); sc = input("Che cosa faccio? [1 calibrazione / 2 baseline / 3 tutto]  (Invio = 3): ").strip() or "3"; t0 = time.time()
    pool = Pool(N_PROC)
    if sc in ("1", "3"):
        res = esegui(list(FT1_POLICIES), SEED_CAL, pool); scrivi(res, percorsi.OUTPUT / "baseline_calibrazione.csv")
        m = {n: np.mean(list(d.values())) for n, d in res.items()}; best = min(m, key=m.get)
        tabella(res, REGOLA, "calibrazione (seed 900-929)"); print(f"\nLa migliore in calibrazione e' {best}; REGOLA in config_ft1.py = {REGOLA}  ->  {'coerente' if best == REGOLA else 'DIVERSA: aggiorna config_ft1.py (e rilancia il baseline)'}")
    if sc in ("2", "3"):
        for nome, seeds in (("val", SEED_VAL), ("test", SEED_TEST)):
            res = esegui([REGOLA, "PLAN 1 passo", "PLAN 2 passi"], seeds, pool)
            pc = ppo_costi(seeds)
            if pc: res["PPO (ppo_ft1_best_r0)"] = pc
            scrivi(res, percorsi.OUTPUT / f"baseline_{nome}.csv"); tabella(res, REGOLA, f"{'validazione 1000-1029' if nome == 'val' else 'TEST FRESCO 9000-9099'}")
    print(f"\n(fatto in {time.time() - t0:.0f} s; file in {percorsi.OUTPUT})")
