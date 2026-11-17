"""
Valutazione di checkpoint LightZero su FT1 con ripetizioni, riprendibile, confronto appaiato con la regola di riferimento di FT1 (stesso ambiente, stessi seed).
Uso (da FT1/, PYTHONPATH=radice:FT1, MZ_SIMS=100):
    python muzero/mz_eval_ft1.py <nome_run> <kind> <iterazioni separate da virgola | last> [repliche=3] [val|test]
val = semi 1000-1029; test = semi 9000-9099 (FRESCHI: usare UNA volta, sull'ultimo checkpoint).
Scrive FT1/runs/<nome_run>/eval_ft1_<val|test>.csv (resume) e stampa la tabella. La ricerca C++ non e' ripetibile: per questo le repliche.
"""
import os, sys, csv, glob, random
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from ding.config import compile_config
from ding.policy import create_policy
_F = Path(__file__).resolve().parent; sys.path[:0] = [str(_F.parent.parent), str(_F.parent.parent / "muzero"), str(_F.parent), str(_F)]      # radice (hpcpower), radice/muzero (slim_aux), FT1 (ft1lib), FT1/muzero
from mz_config_ft1 import make_cfg_ft, FT0_KW
from ft1lib.factory_ft1 import make_env
from ft1lib.policies_ft1 import FT1_POLICIES
from ft1lib.config_ft1 import REGOLA, SEED_VAL, SEED_TEST
RULE = FT1_POLICIES[REGOLA]
REF = {"val": "vedi output/baseline_*.csv (regola, PLAN, PPO)", "test": "", "fresh": "vedi output/baseline_*.csv"}
PPO_FRESH = float(os.environ.get("FT1_PPO_TEST", "1.8758"))                # costo di PPO sul test fresco 9000-9099 (output/baseline_test.csv), per la frazione di guadagno (FT1_PPO_TEST per cambiarlo)


class MZFTAgent:
    def __init__(self, ckpt, kind):
        cfg, create = make_cfg_ft(kind)
        cfg.policy.cuda = False; cfg.policy.device = "cpu"
        cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
        from slim_aux import dims_from_ckpt                                  # ricostruisce le dimensioni delle reti ausiliarie dal checkpoint (vecchio o ridotto)
        d = dims_from_ckpt(ckpt)
        if d: cfg.policy.model.update(d)
        self.policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval"])
        self.policy.learn_mode.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def act(self, env):
        obs = torch.tensor(env._build_observation()[None], dtype=torch.float32)
        mask = env.action_masks().astype(np.int8)
        with torch.no_grad():
            out = self.policy.eval_mode.forward(obs, [mask], to_play=[-1], ready_env_id=np.array([0]), timestep=[0])
        return int(out[0]["action"])


def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def episode(act, s):
    seed_all(s); e = make_env(dict(FT0_KW)); e.reset(seed=s); cost = 0.0; n = 0
    while True:
        _, r, te, tr, _ = e.step(int(act(e))); cost -= r / 3600; n += 1
        if te or tr: return cost, n
        assert n < 1000


def boot(d, B=20000):
    rng = np.random.default_rng(20261108); bt = rng.integers(0, len(d), (B, len(d))); m = d[bt].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)


if __name__ == "__main__":
    run, kind, its = sys.argv[1], sys.argv[2], sys.argv[3]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    which = sys.argv[5] if len(sys.argv) > 5 else "val"
    seeds = list(SEED_VAL) if which == "val" else list(SEED_TEST)               # test/fresh = 9000-9099, seed mai usati (test finale FT1)
    seeds = seeds[:int(os.environ.get("MZ_EVAL_N", len(seeds)))]                  # MZ_EVAL_N=3: solo i primi 3 seed (prova rapida)
    root = Path(__file__).resolve().parent.parent; rdir = root / "runs" / run          # FT1/runs/<nome_run>
    cks = {int(Path(p).name.split("_")[1].split(".")[0]): p for p in glob.glob(str(rdir / "data_mz" / "*" / "ckpt" / "iteration_*.pth.tar"))}
    if not cks: sys.exit(f"nessun checkpoint in {rdir}")
    want = [max(cks)] if its == "last" else [(max(cks) if x == "last" else int(x)) for x in its.split(",")]
    f = rdir / f"eval_ft1_{which}.csv"; done = {}
    if f.exists():
        with f.open() as fh:
            for r in csv.DictReader(fh): done[(r["who"], int(r["replica"]), int(r["seed"]))] = float(r["cost"])
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["who", "replica", "seed", "cost", "steps"])
        if new: w.writeheader()
        for s in seeds:
            if ("regola", 0, s) not in done:
                c, n = episode(RULE, s); w.writerow(dict(who="regola", replica=0, seed=s, cost=c, steps=n)); fh.flush(); done[("regola", 0, s)] = c
        for it in want:
            if it not in cks:
                near = min(cks, key=lambda k: abs(k - it)); print(f"iterazione {it} assente, uso {near}"); it = near
            who = f"it{it}"; agent = MZFTAgent(cks[it], kind)
            for rep in range(reps):
                for s in seeds:
                    if (who, rep, s) in done: continue
                    c, n = episode(agent.act, s); w.writerow(dict(who=who, replica=rep, seed=s, cost=c, steps=n)); fh.flush(); done[(who, rep, s)] = c
                    print("EVAL", who, rep, s, round(c, 4), n, flush=True)
    reg = np.array([done[("regola", 0, s)] for s in seeds]); out = [f"run {run} ({kind}), {which}, {len(seeds)} seed, {reps} repliche | regola {REGOLA} {reg.mean():.4f} h | rif.: {REF[which]}"]
    for it in sorted({int(k[0][2:]) for k in done if k[0] != "regola"}):
        who = f"it{it}"; cs = np.array([[done[(who, r, s)] for s in seeds] for r in range(reps) if (who, r, seeds[0]) in done and all((who, r, s) in done for s in seeds)])
        if not len(cs): continue
        m = cs.mean(0); d = m - reg; lo, hi = boot(d)
        out.append(f"  it {it:6d}: costo {m.mean():.4f} h | vs regola {100*d.mean()/reg.mean():+.1f}% IC95 [{100*lo/reg.mean():+.1f},{100*hi/reg.mean():+.1f}] | seed migliori della regola {(d<0).sum()}/{len(seeds)} | ({len(cs)} repliche)")
    if which == "fresh":
        for it in sorted({int(k[0][2:]) for k in done if k[0] != "regola"}):
            who = f"it{it}"
            if all((who, r, s) in done for r in range(reps) for s in seeds):
                m = np.mean([[done[(who, r, s)] for s in seeds] for r in range(reps)], 0).mean(); out.append(f"  it {it}: frazione del guadagno PPO chiuso {(reg.mean()-m)/(reg.mean()-PPO_FRESH):.2f} (promettente >=0.50; successo >=0.70)")
    txt = "\n".join(out); print(txt); (rdir / f"eval_ft1_{which}_riassunto.txt").write_text(txt + "\n")
