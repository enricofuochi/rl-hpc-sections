"""PPO from scratch on FT1: same protocol as FT0 (preset 'mia': net [64,64], lr 3e-4, gamma 0.999, n_steps 512 x 8 env, batch 256, ent 1e-2; raw reward / 1500; 300000 steps;
validation every 25000 steps; checkpoint chosen by the 3-point moving average on validation). Reference in validation: the FT1 rule; floor: random legal actions.
Usage:  python FT1/train_ppo.py [seed=0] [steps=300000]
Writes FT1/runs/ppo_ft1/ppo_ft1_best_r<seed>.zip (+ _obsrms.pkl) and the history ppo_ft1_r<seed>_history.csv.  ~4-10 minutes for 300000 steps."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import os, numpy as np
import hpcpower.ppo as P
from ft1lib.config_ft1 import KW, REGOLA, SEED_VAL
from ft1lib.factory_ft1 import make_env
from ft1lib.policies_ft1 import FT1_POLICIES

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
chunk = int(os.environ.get("FT1_PPO_CHUNK", 25_000))
rng = np.random.default_rng(0)
reference, floor = FT1_POLICIES[REGOLA], (lambda e: int(rng.choice(np.flatnonzero(e.action_masks()))))
out = percorsi.RUNS / "ppo_ft1"; out.mkdir(parents=True, exist_ok=True); os.chdir(out)
agent, hist = P.train_agent(make_env, dict(KW), reference, floor, val_seeds=list(SEED_VAL), run_seed=seed, total_steps=steps, chunk=chunk, reward_scale=1500.0, out_prefix="ppo_ft1")
agent.save(f"ppo_ft1_best_r{seed}"); hist.to_csv(f"ppo_ft1_r{seed}_history.csv"); print("DONE FT1", seed, flush=True)
