"""Tracce per decisione degli episodi FT0 (stesso formato per regole, PPO e MuZero).
play(e, who, queries) gioca un episodio gia' resettato. who(e) -> (azione, info). queries = {nome: f(e) -> (azione, info)} sono interrogate su ogni stato visitato SENZA agire
(servono per l'accordo tra politiche e per il valore di una politica sugli stati dell'altra). Solo numpy + hpcpower."""
import numpy as np
from hpcpower.resid import costo_residuo_h
from hpcpower.policies_ft import FT_POLICIES
PMAX = 12000.0; RULE = "ft_sjf_th80"

def play(e, who, queries=None, rule_name=RULE):
    rule = FT_POLICIES[rule_name]; queries = queries or {}; rows = []; ticks = []; jobs = list(e.queue); bg = list(e.running); cum = 0.0
    while True:
        mask = e.action_masks().copy(); a, info = who(e); a = int(a)
        q = {n: f(e) for n, f in queries.items()}
        row = dict(i=len(rows), t=e.t, a=a, mask=mask, nq=len(e.queue), cum=cum, ctg_rule=costo_residuo_h(e, rule), info=info, q=q, free=(e.free(0), e.free(1)))
        if a < 2:
            L = e.levels[a]; D = e.demand(); f0, c0 = e.cost_flow_h, e.cost_en_h
            tick = dict(t=e.t, L=L, D=D, deliv=L * PMAX, used=min(D, L * PMAX), nsys=len(e.queue) + sum(1 for j in e.running if not j.bg), nq=len(e.queue),
                        occ=(e.cap[0] - e.free(0), e.cap[1] - e.free(1)), row=len(rows))
        _, r, te, tr, _ = e.step(a); cum -= r / 3600.0
        if a < 2: tick.update(s=e.last_s, flow_inc=e.cost_flow_h - f0, en_inc=e.cost_en_h - c0); ticks.append(tick)
        rows.append(row)
        if te or tr: break
    for x in rows: x["ctg"] = cum - x["cum"]
    jt = [dict(id=j.id, n=j.n, typ=j.typ, nom=j.nom, arr=j.arr, start=j.start, end=j.end, sec=j.sec) for j in jobs]
    jb = [dict(id=j.id, n=j.n, typ=j.typ, nom=j.nom, arr=-1, start=0, end=j.end if j.end >= 0 else e.t, sec=j.sec) for j in bg]
    return dict(jobs_bg=jb, rows=rows, ticks=ticks, jobs=jt, total=cum, flow=e.cost_flow_h, energy=e.cost_en_h, makespan_h=e.t * e.dt / 3600.0, dt=e.dt, levels=e.levels)

def rule_who(name):
    pol = FT_POLICIES[name]; return lambda e: (int(pol(e)), {})

def _obs(ag, e):
    o = e._build_observation()
    if ag.obs_rms is not None: o = np.clip((o - ag.obs_rms.mean) / np.sqrt(ag.obs_rms.var + 1e-8), -ag.clip, ag.clip).astype(np.float32)
    return o

def ppo_who(ag, reward_scale=1500.0):
    """Azione deterministica come Agent.act; in piu' valore (in ore di costo residuo: -V*scala/3600, vale con gamma=1) e probabilita' mascherate."""
    import torch
    gamma = float(ag.model.gamma)
    def f(e):
        a = ag.act(e); o = _obs(ag, e); mask = e.action_masks()
        with torch.no_grad():
            t = torch.as_tensor(o[None]); v = float(ag.model.policy.predict_values(t).item())
            p = ag.model.policy.get_distribution(t, action_masks=mask[None]).distribution.probs[0].numpy()
        return a, dict(ctg_pred=-v * reward_scale / 3600.0, probs=p.astype(np.float32), gamma=gamma)
    return f

def load_ppo(path):
    """Carica un MaskablePPO salvato su un'altra versione di numpy/python: gli oggetti serializzati con cloudpickle (spazi, ultimo stato, schedule) NON si deserializzano
    tra numpy 2 e numpy 1; per la sola inferenza servono solo i pesi, quindi li sostituiamo (custom_objects). Spazi di FT0: osservazione 146 valori in [0,1], 14 azioni."""
    import pickle
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    import hpcpower.ppo as P
    co = dict(observation_space=spaces.Box(0.0, 1.0, (146,), np.float32), action_space=spaces.Discrete(14), _last_obs=None, _last_episode_starts=None, lr_schedule=lambda _: 3e-4, clip_range=lambda _: 0.2)
    with open(path + "_obsrms.pkl", "rb") as f: rms = pickle.load(f)
    return P.Agent(MaskablePPO.load(path, custom_objects=co, device="cpu"), rms)
