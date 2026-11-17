"""FT1: FT0 con un controllo della potenza che ha MEMORIA.

In FT0 ogni tick si sceglie il livello L in {1.0, 0.7} in modo indipendente. In FT1 il livello e' uno STATO (L corrente, parte da 1.0) e la decisione di avanzamento e' una MOSSA:
    azione 0 = L - delta,  azione 1 = L (resta),  azione 2 = L + delta     (delta = 0.1, L in [lmin = 0.5, 1.0]; le mosse fuori range sono mascherate)
    azione 3 + 2c + s = avvia il job piu' vecchio della classe c = 2*indice_taglia + tipo sulla sezione s            (15 azioni in tutto; in FT0 erano 14)
Il nuovo L vale per il tick che si sta chiudendo. Il resto e' identico a FT0: fisica, costo per tick (n_sys*dt_h + w*L*dt_h)/N, nessuna attesa (si puo' avanzare solo se nessun piazzamento e' legale),
potenza erogata P_del = L*P_max pagata tutta, quota s = min(1, P_del/D). Osservazione: stessi 146 numeri di FT0, con la casella 5 (in FT0: livello precedente binario) sostituita da L corrente (continuo).
Additivo: i file di hpcpower/ non sono toccati (si importa FT0Env e si ridefiniscono solo azioni, maschere, avanzamento e osservazione)."""
import numpy as np
from gymnasium import spaces
from hpcpower.env_ft0 import FT0Env
from hpcpower.env_ft import PMAX, NOMINAL_W, PMIN_FRAC, GAMMA, speed

MOSSE = (-1, 0, 1)
NOMI_MOSSE = ("L-", "L=", "L+")


class FT1Env(FT0Env):
    def __init__(self, delta=0.1, lmin=0.5, **kw):
        kw.pop("levels", None)
        super().__init__(**kw)
        self.delta, self.lmin, self.moves, self.na = float(delta), float(lmin), MOSSE, len(MOSSE)
        self.n_act = self.na + self.n_cls * self.n_sec
        self.action_space = spaces.Discrete(self.n_act)
        self.curL = 1.0

    # ------------------------------------------------------------------ livello
    def new_L(self, i): return round(min(1.0, max(self.lmin, self.curL + self.moves[i] * self.delta)), 4)

    def legal_adv(self, i):
        x = round(self.curL + self.moves[i] * self.delta, 4)
        return self.lmin - 1e-9 <= x <= 1.0 + 1e-9

    def reset(self, seed=None, options=None):
        self.curL = 1.0
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------ azioni
    def placements(self): return [a - 2 + self.na for a in super().placements()]      # la base numera i piazzamenti da 2

    def action_masks(self):
        m = np.zeros(self.n_act, dtype=bool); pl = self.placements()
        for a in pl: m[a] = True
        if not pl:                                                          # senza attesa: si avanza solo se non c'e' nessun piazzamento
            for i in range(self.na): m[i] = self.legal_adv(i)
        return m

    def step(self, action):
        a = int(action)
        if a >= self.na:
            c, s = divmod(a - self.na, 2); h = self.heads().get(c)
            if h is None or self.free(s) < h.n: raise ValueError("azione non legale")
            self.queue.remove(h); self._start_job(h, s)
            return self._build_observation(), 0.0, False, False, {"action_mask": self.action_masks()}
        if not self.legal_adv(a): raise ValueError("mossa di L non legale")
        L = self.new_L(a); dt_h = self.dt / 3600.0
        nsys = len(self.queue) + sum(1 for j in self.running if not j.bg)
        cost = (nsys * dt_h + self.w * L * dt_h) / self.N
        self.cost_flow_h += nsys * dt_h / self.N; self.cost_en_h += self.w * L * dt_h / self.N
        D = self.demand(); s = 1.0 if D <= 0 else min(1.0, L * PMAX / D); self.curL, self.last_L, self.last_s = L, L, s
        fin = []
        for j in self.running:
            v = speed(s * j.dem, j.dem, PMIN_FRAC * NOMINAL_W[j.sec], GAMMA[j.typ])
            j.rem -= v * self.dt
            if j.rem <= 1e-6: fin.append(j)
        for j in fin: self.running.remove(j); j.end = self.t + 1
        self.t += 1
        if self.t < len(self.sched): self.queue += self.sched[self.t]; self.sched[self.t] = []
        te = self._terminal(); tr = (not te) and self.t >= self.max_ticks
        self.ret += cost
        return self._build_observation(), -3600.0 * cost, te, tr, {"action_mask": self.action_masks(), "makespan": self.t * self.dt}

    # ------------------------------------------------------------------ osservazione
    def _build_observation(self):
        ob = super()._build_observation()
        ob[5] = self.curL                                                   # L corrente (in FT0: 1.0 oppure 0.7/1.0)
        return ob
