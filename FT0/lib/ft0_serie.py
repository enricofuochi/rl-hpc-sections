"""Serie temporali, metriche globali e metriche per job di un episodio FT0 (energia, makespan, potenza nel tempo, stress, potenza per nodo, curve per job).
ep_series(e, act): e gia' resettato, act(e) -> azione. Solo numpy + hpcpower."""
import numpy as np
from hpcpower.env_ft import speed, PMIN_FRAC, NOMINAL_W, GAMMA
PMAX = 12000.0

def ep_series(e, act):
    jobs = list(e.queue); en = {j.id: 0.0 for j in jobs}; T = []
    while True:
        a = int(act(e))
        if a < 2:
            L = e.levels[a]; run = list(e.running); nodes = sum(j.n for j in run); D = e.demand(); t0 = e.t; nq = len(e.queue)
            _, r, te, tr, _ = e.step(a); s = e.last_s; used = 0.0; st = 0.0
            for j in run:
                p = s * j.dem * j.n; used += p; st += (1.0 - speed(s * j.dem, j.dem, PMIN_FRAC * NOMINAL_W[j.sec], GAMMA[j.typ])) * j.n
                if j.id in en: en[j.id] += p * e.dt / 3.6e6
            T.append((t0, L, D, L * PMAX, used, nodes, st / max(nodes, 1), nq))
        else: _, r, te, tr, _ = e.step(a)
        if te or tr: break
    A = np.array(T, float); dt = e.dt
    jb = np.array([[(j.end - j.arr) * dt / 3600.0, (j.end - j.start) * dt / 3600.0, (j.start - j.arr) * dt / 3600.0, en[j.id], j.typ, j.n, j.sec] for j in jobs])
    return dict(total=e.ret, flow=e.cost_flow_h, energy_term=e.cost_en_h, makespan_h=e.t * dt / 3600.0, E_kWh=A[:, 3].sum() * dt / 3.6e6, E_used_kWh=A[:, 4].sum() * dt / 3.6e6,
                dt=dt, t=A[:, 0], L=A[:, 1], D=A[:, 2], deliv=A[:, 3], used=A[:, 4], nodes=A[:, 5], stress=A[:, 6], nq=A[:, 7], jobs=jb)   # jobs: flusso_h, durata_h, attesa_h, energia_kWh, tipo, taglia, sezione
