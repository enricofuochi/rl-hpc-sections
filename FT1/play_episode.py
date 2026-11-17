"""Play one FT1 episode with the saved players. Minimal entry point: load players -> play the same seed with each -> print costs (and, optionally, the decisions or an HTML replay).

    python FT1/play_episode.py                                   # seed 9000, players: rule ppo mz
    python FT1/play_episode.py --seed 9017 --players rule ppo plan1
    python FT1/play_episode.py --seed 9000 --players rule ppo --log          # decision-by-decision log
    python FT1/play_episode.py --seed 9000 --html episode.html               # side-by-side interactive replay (rule / PPO / MuZero)

Players (--players): rule (L-tracking rule, the reference), l1 (same placement rule with L = 1), rand (uniform over legal actions), plan1 / plan2 (PRIVILEGED planners that copy the exact
simulator state), ppo (MaskablePPO checkpoint), mz (MuZero checkpoint, 100 MCTS simulations; needs torch + LightZero). Rules, planners and PPO need only numpy, gymnasium and sb3-contrib.
Defaults come from config/config_fissa.py (the configuration of the thesis); --ppo / --muzero / --sims override the checkpoints. The episode is fully determined by the seed
(24 jobs at t=0 and the pre-filled cluster), so every player faces exactly the same instance. Fresh test seeds are 9000-9099. MuZero's C++ search is not exactly repeatable across runs."""
import sys, pathlib, argparse, tempfile, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import ft1_config as C, ft1_giocatori as G
from ft1_traccia import play
from ft1lib.factory_ft1 import make_env
from hpcpower.env_ft import SIZES

SECTION = ("fast", "efficient"); KIND = ("compute", "memory"); MOVE = ("L-", "L=", "L+")


def describe(a, na, L_after):
    """Human-readable action: level move (with the resulting L) or job placement (class c = 2 * size index + type; section s)."""
    if a < na: return f"advance one tick, {MOVE[a]} -> L = {L_after:.1f}"
    c, s = divmod(a - na, 2); return f"start the oldest queued {SIZES[c // 2]}-node {KIND[c % 2]} job on the {SECTION[s]} section"


def print_log(name, tr):
    ticks = {t["row"]: t for t in tr["ticks"]}
    print(f"\n--- {name}: decisions (t in minutes, cum = cost accumulated so far in hours) ---")
    for r in tr["rows"]:
        L = ticks[r["i"]]["L"] if r["i"] in ticks else None
        print(f"  t={r['t'] * tr['dt'] / 60:5.0f}  queue {r['nq']:2d}  cum {r['cum']:6.3f}  {describe(r['a'], tr['na'], L)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=9000); ap.add_argument("--players", nargs="+", default=["rule", "ppo", "mz"], choices=list(G.NOMI))
    ap.add_argument("--log", action="store_true", help="print every decision"); ap.add_argument("--html", help="write an interactive replay (rule/ppo/mz players only)")
    ap.add_argument("--ppo", help="PPO checkpoint path without .zip"); ap.add_argument("--muzero", help="MuZero checkpoint (.pth.tar)"); ap.add_argument("--sims", type=int, help="MCTS simulations")
    a = ap.parse_args()
    cfg = C.carica("fissa")
    if a.ppo: cfg["PPO_CHECKPOINT"] = a.ppo
    if a.muzero: cfg["MUZERO_CHECKPOINT"] = a.muzero
    if a.sims: cfg["MZ_SIMULAZIONI"] = a.sims
    kw = C.env_kw(cfg); W = G.costruisci(cfg, a.players); res = {}
    for p in a.players:
        G.seed_all(a.seed); e = make_env(dict(kw)); e.reset(seed=a.seed)
        if hasattr(W[p], "reset"): W[p].reset(a.seed)
        queries = {q: W[q] for q in ("ppo", "mz") if a.html and q in W and q != p} if p in ("rule", "ppo", "mz") else {}      # cross-queries feed the replay (PPO probabilities, MuZero visits)
        res[p] = {a.seed: play(e, W[p], queries, rule_name=cfg["REGOLA"])}
    ref = res.get("rule", {}).get(a.seed)
    print(f"\nSeed {a.seed}  (cost = mean job flow time + w * energy term, in hours; lower is better)")
    print(f"  {'player':18s} {'cost':>7s} {'flow':>7s} {'energy':>7s} {'makespan':>9s} {'decisions':>10s}" + (f" {'vs rule':>8s}" if ref else ""))
    for p in a.players:
        t = res[p][a.seed]
        print(f"  {G.NOMI[p]:18s} {t['total']:7.3f} {t['flow']:7.3f} {t['energy']:7.3f} {t['makespan_h']:8.2f}h {len(t['rows']):10d}" + (f" {100 * (t['total'] / ref['total'] - 1):+7.1f}%" if ref else ""))
    if a.log:
        for p in a.players: print_log(G.NOMI[p], res[p][a.seed])
    if a.html:
        from ft1_viewer import genera
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f: pickle.dump(dict(res=res, meta=dict(regola=cfg["REGOLA"], env_kw=kw)), f)
        out, mb = genera(f.name, a.html); print(f"\nWrote {out} ({mb:.1f} MB); open it in a browser.")


if __name__ == "__main__":
    main()
