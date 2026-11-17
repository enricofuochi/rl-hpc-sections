"""Controlli di FT1 (play): l'ambiente nuovo e' coerente? ~30 secondi. Se tutto e' 'OK' si puo' procedere con baseline (01_baseline.py) e training.
  1. forma: 15 azioni, 146 osservazioni nel range [0,1], maschere (mosse di L fuori range vietate, avanzamento solo senza piazzamenti)
  2. equivalenza con FT0: con delta=0.3 e L minimo 0.7 (cioe' L in {1.0, 0.7}) e la regola di soglia di FT0, FT1 da' ESATTAMENTE gli stessi costi di FT0 (ft_sjf_th80)
  3. reward di scarto (MuZero): la somma dei reward modificati = scala * (costo regola - costo agente); con le azioni della regola ogni reward vale 0
  4. wrapper LightZero (solo se 'ding' e' installato, cioe' sul Mac nel .venv)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import os, numpy as np
from ft1lib.factory_ft1 import make_env
from ft1lib.config_ft1 import KW, REGOLA
from ft1lib.policies_ft1 import FT1_POLICIES, regola_th
from hpcpower.resid import costo_residuo_h

ok_tot = True
def check(nome, cond, extra=""):
    global ok_tot; ok_tot &= bool(cond); print(f"  [{'OK' if cond else 'FAIL'}] {nome} {extra}", flush=True)

def costo(e, pol, s):
    e.reset(seed=s); T = 0.0; n = 0
    while True:
        _, r, te, tr, _ = e.step(int(pol(e))); T -= r; n += 1
        if te or tr: return T / 3600.0, n

print("1. forma e maschere")
e = make_env(KW); o, info = e.reset(seed=1000)
check("15 azioni", e.n_actions == 15, f"(n_act={e.n_actions})"); check("osservazione 146 numeri in [0,1]", o.shape == (146,) and o.min() >= 0 and o.max() <= 1)
check("L iniziale 1.0 nella casella 5 dell'osservazione", abs(o[5] - 1.0) < 1e-6)
rng = np.random.default_rng(1); viol = 0; mosse_fuori = 0; n_adv = 0; Lmin_visto, Lmax_visto = 1.0, 0.0
for s in range(1000, 1010):
    o, info = e.reset(seed=s)
    while True:
        m = e.action_masks(); pl = e.placements()
        if pl: viol += int(m[:3].any())                                     # con un piazzamento legale non si puo' avanzare
        else: viol += int(not m[:3].any()); mosse_fuori += int(m[0] and e.curL - e.delta < e.lmin - 1e-9) + int(m[2] and e.curL + e.delta > 1 + 1e-9)
        a = int(rng.choice(np.flatnonzero(m))); n_adv += int(a < 3)
        o, r, te, tr, info = e.step(a); Lmin_visto = min(Lmin_visto, e.curL); Lmax_visto = max(Lmax_visto, e.curL)
        assert abs(o[5] - e.curL) < 1e-6 and np.array_equal(info["action_mask"], e.action_masks())
        if te or tr: break
check("maschere coerenti (avanzamento solo senza piazzamenti, almeno una mossa legale)", viol == 0)
check("nessuna mossa fuori dal range [0.5, 1.0]", mosse_fuori == 0 and Lmin_visto >= 0.5 - 1e-9 and Lmax_visto <= 1.0 + 1e-9, f"(L visto: {Lmin_visto:.1f} .. {Lmax_visto:.1f})")
c, n = costo(make_env(KW), FT1_POLICIES[REGOLA], 1000); check("episodio della regola: ~51 decisioni", 40 <= n <= 70, f"(costo {c:.3f} h, {n} decisioni)")

print("2. equivalenza con FT0 (delta=0.3, L minimo 0.7, regola di soglia)")
from hpcpower.factory import make_env as make_env0
from hpcpower.policies_ft import FT_POLICIES
kw1 = dict(KW); kw1.update(delta=0.3, lmin=0.7)
kw0 = dict(env="ft0", n_jobs=24, dt=600.0, w=5.7, pf=(0.70, 0.80), dur_med=1800.0, sigma=1.0, max_ticks=400, levels=(1.0, 0.7), xobs=True, qk=24)
e1, e0 = make_env(kw1), make_env0(kw0)
dmax = max(abs(costo(e1, regola_th(), s)[0] - costo(e0, FT_POLICIES["ft_sjf_th80"], s)[0]) for s in range(7000, 7030))
check("FT1(delta=0.3, lmin=0.7) = FT0 su 30 seed", dmax < 1e-9, f"(differenza massima {dmax:.1e} h)")

print("3. reward di scarto dalla regola (scala 100)")
R = FT1_POLICIES[REGOLA]; scala = 100.0; err = 0.0; rule_nz = 0.0
for s in (1000, 1001, 1002):
    e = make_env(KW); e.reset(seed=s); phi = -costo_residuo_h(e, R); c0 = -phi; tot = 0.0; ret = 0.0
    rng = np.random.default_rng(s)
    while True:
        m = e.action_masks(); a = int(R(e)) if rng.random() < 0.7 else int(rng.choice(np.flatnonzero(m)))
        _, r, te, tr, _ = e.step(a); rew = r / 3600.0; done = te or tr; phi1 = 0.0 if done else -costo_residuo_h(e, R)
        tot += scala * (rew + phi1 - phi); ret -= rew; phi = phi1
        if done: break
    err = max(err, abs(tot - scala * (c0 - ret)))
    e = make_env(KW); e.reset(seed=s); phi = -costo_residuo_h(e, R)
    while True:
        _, r, te, tr, _ = e.step(int(R(e))); done = te or tr; phi1 = 0.0 if done else -costo_residuo_h(e, R); rule_nz = max(rule_nz, abs(scala * (r / 3600.0 + phi1 - phi))); phi = phi1
        if done: break
check("somma dei reward di scarto = scala * (costo regola - costo agente)", err < 1e-6, f"(errore massimo {err:.1e})")
check("con le azioni della regola ogni reward di scarto vale 0", rule_nz < 1e-6, f"(massimo {rule_nz:.1e})")

print("4. wrapper LightZero")
try: import ding  # noqa: F401
except ModuleNotFoundError: print("  (salto: 'ding' non e' installato in questo interprete; il controllo vero e' la prova rapida:  bash FT1/train_muzero.sh ft1_smoke 0 smoke)")
else:
    os.environ["MZ_RESID"] = REGOLA; os.environ["MZ_RESID_SCALE"] = "100"
    from ft1lib.lz_env_ft1 import HPCFT1LZEnv
    w = HPCFT1LZEnv(dict(hpc_kw=dict(KW))); ob = w.reset(); n = 0; tot = 0.0
    while True:
        a = int(R(w._env)); ts = w.step(a); n += 1; tot += ts.reward
        if ts.done: break
    check("wrapper: episodio con la regola, reward totale 0", abs(tot) < 1e-6 and ob["observation"].shape == (146,) and ob["action_mask"].shape == (15,), f"({n} passi, reward {tot:.1e})")
print("\nTUTTO OK" if ok_tot else "\nCI SONO CONTROLLI FALLITI")
