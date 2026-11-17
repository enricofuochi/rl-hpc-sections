"""Caricamento dei 'giocatori' di FT0: regola scritta, regola con attesa, casuale, PPO, MuZero.
Un giocatore e' una funzione who(env) -> (azione, info) (info = dettagli per le tracce: probabilita' PPO, visite MuZero...)."""
import os, random
from pathlib import Path
import numpy as np
import percorsi
from ft0_traccia import rule_who, ppo_who, load_ppo

NOMI = dict(rule="Rule (no waiting)", rulew="Rule (waiting)", rand="Random", ppo="PPO", mz="MuZero")
ATTESA = {"rule": False, "rulew": True, "rand": False, "ppo": False, "mz": False}      # l'ambiente deve consentire l'attesa (allow_wait)?


def percorso(rel):
    p = Path(rel); return p if p.is_absolute() else percorsi.FT0 / p


def seed_all(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s)
    except ImportError: pass


def regola_con_attesa(nome): return nome.replace("ft_", "ftW_", 1)


def random_who():
    """Azione uniforme tra quelle legali (seed+7 per episodio, come nei riferimenti della tesi). Chiamare f.reset(seed) prima di ogni episodio."""
    st = {}
    def f(e): return int(st["rng"].choice(np.flatnonzero(e.action_masks()))), {}
    f.reset = lambda s: st.update(rng=np.random.default_rng(s + 7)); return f


def carica_ppo(cfg):
    base = str(percorso(cfg["PPO_CHECKPOINT"]))
    if not Path(base + ".zip").exists(): raise SystemExit(f"Non trovo il checkpoint PPO: {base}.zip")
    ag = load_ppo(base); print(f"PPO caricato ({Path(base).name}), gamma {ag.model.gamma}", flush=True); return ag


def carica_muzero(cfg):
    """Restituisce (agente LightZero, who). Needs an environment with torch + LightZero."""
    os.environ["MZ_SIMS"] = str(cfg["MZ_SIMULAZIONI"])
    import torch; torch.set_num_threads(1)
    try: from mz_eval_ft import MZFTAgent
    except ModuleNotFoundError as ex:
        import sys
        raise SystemExit(f"Missing module '{ex.name}': MuZero needs an environment with torch + LightZero (see README). Interpreter in use: {sys.executable}")
    ck = percorso(cfg["MUZERO_CHECKPOINT"])
    if not ck.exists(): raise SystemExit(f"Non trovo il checkpoint MuZero: {ck}")
    mza = MZFTAgent(str(ck), "muzero"); print(f"MuZero caricato ({ck.name}), {cfg['MZ_SIMULAZIONI']} simulazioni", flush=True)

    def who(e):
        obs = torch.tensor(e._build_observation()[None], dtype=torch.float32); mask = e.action_masks().astype(np.int8)
        with torch.no_grad(): o = mza.policy.eval_mode.forward(obs, [mask], to_play=[-1], ready_env_id=np.array([0]), timestep=[0])[0]
        vc = np.asarray(o.get("visit_count_distributions", []), dtype=float); n_act = len(mask); v = np.zeros(n_act); legal = np.flatnonzero(mask)
        if len(vc) == n_act: v = vc
        elif len(vc) == len(legal): v[legal] = vc
        v = v / max(v.sum(), 1e-9)
        return int(o["action"]), dict(value=float(o.get("searched_value", np.nan)), pred_value=float(o.get("predicted_value", np.nan)), visits=v.astype(np.float32))
    return mza, who


def costruisci(cfg, players):
    """Dict nome -> who per i giocatori richiesti (carica PPO/MuZero solo se servono)."""
    W = {}
    if "rule" in players: W["rule"] = rule_who(cfg["REGOLA"])
    if "rulew" in players: W["rulew"] = rule_who(regola_con_attesa(cfg["REGOLA"]))
    if "rand" in players: W["rand"] = random_who()
    if "ppo" in players: W["ppo"] = ppo_who(carica_ppo(cfg))
    if "mz" in players: W["mz"] = carica_muzero(cfg)[1]
    return W
