"""Reproducibility check for FT1: mean episode cost of the saved players on the fresh test seeds (9000-9099 by default), compared with the values in the README / thesis.

    python FT1/reproduce_results.py                         # rule, PLAN 1 step, PPO on 100 test seeds (about 1-2 min)
    python FT1/reproduce_results.py --players rule ppo      # a subset
    python FT1/reproduce_results.py --players rule ppo mz --seeds 9000 9010   # MuZero (needs torch + LightZero), 10 seeds; slow, search is not exactly repeatable
    python FT1/reproduce_results.py --procs 4               # parallel processes (rule / plan / ppo only)

Cost = mean job flow time + w * energy term, in hours, averaged over the seeds. Rules, PLAN and PPO are deterministic: they must match the reference to 4 decimals (exit code 1 otherwise).
MuZero has no tolerance check: its C++ search differs slightly between runs, so compare with the thesis table (mean over the same seeds). Writes output/reproduce_results.csv (cost per seed and player)."""
import sys, pathlib, argparse, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
from multiprocessing import Pool
import ft1_config as C, ft1_giocatori as G
from ft1lib.factory_ft1 import make_env

EXPECTED = dict(rule=2.3061, plan1=1.8530, ppo=1.8758)      # mean cost (h) on seeds 9000-9099, configuration of the thesis (config/config_fissa.py)
_W = {}


def episode_cost(who, seed, kw):
    G.seed_all(seed); e = make_env(dict(kw)); e.reset(seed=seed)
    if hasattr(who, "reset"): who.reset(seed)
    cost = 0.0
    while True:
        a, _ = who(e); _, r, te, tr, _ = e.step(int(a)); cost -= r / 3600.0
        if te or tr: return cost


def _init(cfg, players):
    _W.update(G.costruisci(cfg, players)); _W["kw"] = C.env_kw(cfg)


def _job(args):
    p, s = args; return p, s, episode_cost(_W[p], s, _W["kw"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--players", nargs="+", default=["rule", "plan1", "ppo"], choices=["rule", "l1", "rand", "plan1", "plan2", "ppo", "mz"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[9000, 9100], metavar=("FIRST", "LAST_EXCLUDED")); ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--tol", type=float, default=5e-5, help="tolerance on the mean (the reference values have 4 decimals)")
    a = ap.parse_args(); cfg = C.carica("fissa"); seeds = list(range(*a.seeds)); full = a.seeds == [9000, 9100]
    jobs = [(p, s) for p in a.players for s in seeds]
    if a.procs > 1 and "mz" not in a.players:
        with Pool(a.procs, initializer=_init, initargs=(cfg, a.players)) as pool: out = pool.map(_job, jobs, chunksize=4)
    else:
        _init(cfg, a.players); out = [_job(j) for j in jobs]
    res = {p: {} for p in a.players}
    for p, s, c in out: res[p][s] = c
    print(f"\nFT1, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} episodes), mean cost in hours")
    print(f"  {'player':18s} {'mean':>8s} {'reference':>10s} {'diff':>9s}  check")
    ok = True
    for p in a.players:
        m = float(np.mean(list(res[p].values()))); ref = EXPECTED.get(p) if full else None
        if ref is None: print(f"  {G.NOMI[p]:18s} {m:8.4f} {'-':>10s} {'-':>9s}  {'(no reference: non-deterministic or not in the thesis table)' if p == 'mz' else '(reference only for the full test set)' if p in EXPECTED else ''}"); continue
        good = abs(round(m, 4) - ref) <= a.tol + 1e-9; ok &= good
        print(f"  {G.NOMI[p]:18s} {m:8.4f} {ref:10.4f} {m - ref:+9.4f}  {'OK' if good else 'DIFFERENT'}")
    if "rule" in res:
        print("\n  paired difference vs rule:", ", ".join(f"{G.NOMI[p]} {100 * (np.mean([res[p][s] - res['rule'][s] for s in seeds]) / np.mean(list(res['rule'].values()))):+.1f}%" for p in a.players if p != "rule"))
    percorsi.OUTPUT.mkdir(exist_ok=True); f = percorsi.OUTPUT / "reproduce_results.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["seed"] + a.players); [w.writerow([s] + [f"{res[p][s]:.6f}" for p in a.players]) for s in seeds]
    print(f"\nPer-seed costs written to {f}"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
