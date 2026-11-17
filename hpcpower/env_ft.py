"""
FTEnv: ambiente a TEMPO FISSO (tick di dt secondi) per scheduling + livello di erogazione della potenza, 6 nodi veloci + 10 efficienti (come Z3H10MM).

Scelte di progetto (dichiarate a priori, non tarate su un agente):
  * PASSO FISSO. Un episodio e' una successione di tick. Dentro un tick l'agente fa piazzamenti (sotto-passi a tempo zero) e poi chiude il tick con un'azione di AVANZAMENTO che sceglie
    il livello di erogazione L in {1.0, 0.7} per il tick. I job che arrivano nello stesso tick sono contemporanei (disponibili all'inizio del tick).
  * ARRIVI + CLUSTER PRE-RIEMPITO. All'inizio il cluster e' occupato al 70-80% da job "di fondo" (veloce: compute, efficiente: memory; avanzamento gia' compiuto ~ U(0.1, 0.9));
    i job dell'episodio arrivano con conteggio Poisson(lam) per tick durante T_a tick; poi si svuota. L'episodio termina quando non c'e' piu' nulla.
  * JOB: taglia (1,2,4) con pesi (0.40,0.35,0.25); tipo compute/memory con probabilita' dipendenti dalla taglia ((0.3,0.7),(0.5,0.5),(0.7,0.3) per compute/memory:
    qui P(compute) = (0.3, 0.5, 0.7) per taglia 1/2/4); durata nominale = lognormale (mediana 1800 s, sigma 1) x n / 2.1, ritagliata a [dt, 14400] s (durata legata alla dimensione); tipo DICHIARATO (noto all'invio)
    (Stadio A: l'ambiente e' deterministico e completamente osservabile a parte gli arrivi futuri).
  * FISICA (come Z3H10MM): sezione 0 (6 nodi, potenza richiesta x 1000/850, compute x4.0) e sezione 1 (10 nodi, x 600/850, compute x0.6); memory x1.0 su entrambe;
    richiesta per nodo = base x scala sezione (base U[0.87,1.0] x 850 W); lavoro = durata_nominale / moltiplicatore (secondi a velocita' piena); gamma compute 1.0, memory 0.3;
    velocita' = speed_factor(assegnata, richiesta, p_min = 0.2353 x potenza nominale del nodo, gamma): 0 sotto p_min, 1 sopra la richiesta, ((a - p_min)/(p - p_min))^gamma in mezzo.
  * EROGAZIONE: P_del = L x P_max (P_max = 12000 W = 6x1000 + 10x600). D = somma delle richieste dei job in esecuzione. Quota s = min(1, P_del / D); ogni nodo riceve s x richiesta.
    La potenza erogata si PAGA TUTTA (anche quella non usata): lo spreco per eccesso esiste, la riduzione costa rallentamento. Nessuna media mobile.
  * COMPLETAMENTI: un nodo si libera alla fine del tick in cui il lavoro residuo si azzera (la frazione di tick restante e' persa): niente eventi a meta' tick.
  * COSTO per tick (reward = -costo, in "secondi equivalenti", cioe' ore x 3600): (n_sistema x dt_h + w x L x dt_h) / N, con n_sistema = job dell'episodio in coda o in esecuzione
    all'inizio del tick (i job di fondo non contano nel flusso), N = job dell'episodio. Somma = tempo medio di permanenza per job + w x (ore a piena erogazione equivalenti) / N.
    w e' il peso di un'ora di erogazione piena in ore-job (tarato a priori su semi di calibrazione 900-929: energia = flusso per la regola con L = 1).
  * ATTESA: allow_wait=False (default): l'agente NON puo' avanzare se esiste un piazzamento legale (conserva il lavoro). allow_wait=True: puo' tenere i job in coda.
  * AZIONI (14): 0 = avanza con L=1.0, 1 = avanza con L=0.7, 2 + 2*c + s = avvia il job piu' vecchio della classe c = 2*indice_taglia + tipo (tipo 0 compute, 1 memory) nella sezione s.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

SIZES = (1, 2, 4)
SIZE_W = (0.40, 0.35, 0.25)
P_COMPUTE = (0.3, 0.5, 0.7)
SECTIONS = ((6, 1000.0 / 850.0, (4.0, 1.0)), (10, 600.0 / 850.0, (0.6, 1.0)))       # (nodi, scala potenza richiesta, moltiplicatore di velocita' per tipo (compute, memory))
NOMINAL_W = (1000.0, 600.0)
PMIN_FRAC = 0.2353
GAMMA = (1.0, 0.3)
LEVELS = (1.0, 0.7)
PMAX = 12000.0


def speed(a, p, pmin, g):
    if a <= pmin: return 0.0
    if a >= p: return 1.0
    r = (a - pmin) / (p - pmin)
    return r if g == 1.0 else r ** g


class Job:
    __slots__ = ("id", "n", "typ", "nom", "base", "arr", "sec", "rem", "dem", "bg", "start", "end", "cls")

    def __init__(self, id, n, typ, nom, base, arr, bg=False):
        self.id, self.n, self.typ, self.nom, self.base, self.arr, self.bg = id, n, typ, nom, base, arr, bg
        self.sec = -1; self.rem = 0.0; self.dem = 0.0; self.start = self.end = -1; self.cls = 2 * SIZES.index(n) + typ


class FTEnv(gym.Env):
    eval_metric = "cost"

    def __init__(self, dt=300.0, T_a=36, lam=1.0, w=1.0, allow_wait=False, pf=(0.70, 0.80), dur_med=1800.0, sigma=1.0, max_ticks=400, levels=LEVELS, xobs=False, qk=16, **kw):
        super().__init__()
        self.dt, self.T_a, self.lam, self.w, self.allow_wait = float(dt), int(T_a), float(lam), float(w), bool(allow_wait)
        self.pf, self.dur_med, self.sigma, self.max_ticks, self.levels = tuple(pf), float(dur_med), float(sigma), int(max_ticks), tuple(levels)
        self.n_cls, self.n_sec = 6, 2
        self.n_act = len(self.levels) + self.n_cls * self.n_sec
        self.xobs, self.qk = bool(xobs), int(qk)
        self.obs_dim = 7 + self.n_cls * 3 + self.n_sec * 4 + (self.qk * 4 + 1 + 16 if self.xobs else 0)
        self.action_space = spaces.Discrete(self.n_act)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.cap = [s[0] for s in SECTIONS]
        self._uid = 0

    # ------------------------------------------------------------------ generazione
    def _job(self, rng, tick, bg=False):
        i = int(rng.choice(3, p=SIZE_W)); n = SIZES[i]; typ = 0 if rng.random() < P_COMPUTE[i] else 1
        nom = float(np.clip(rng.lognormal(math.log(self.dur_med), self.sigma) * n / 2.1, self.dt, 14400.0))
        base = float(int(rng.uniform(0.87 * 850.0, 850.0)))
        self._uid += 1
        return Job(self._uid, n, typ, nom, base, tick, bg)

    def _start_job(self, j, s):
        j.sec = s; j.dem = j.base * SECTIONS[s][1]; j.rem = j.nom / SECTIONS[s][2][j.typ]; j.start = self.t
        self.running.append(j)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(int(seed) if seed is not None else int(self.np_random.integers(2 ** 31)))
        self._uid = 0; self.t = 0; self.queue = []; self.running = []; self.done_ids = 0; self.last_L = 1.0; self.last_s = 1.0
        self.cost_flow_h = 0.0; self.cost_en_h = 0.0; self.ret = 0.0
        self.sched = []
        for k in range(self.T_a):
            self.sched.append([self._job(rng, k) for _ in range(int(rng.poisson(self.lam)))])
        self.N = max(1, sum(len(x) for x in self.sched))
        u = float(rng.uniform(*self.pf)) if self.pf[1] > 0 else 0.0
        for s, tp in ((0, 0), (1, 1)):                                              # fondo: veloce compute, efficiente memory
            target = int(round(u * self.cap[s])); filled = 0; tries = 0
            while filled < target and tries < 200:
                tries += 1; j = self._job(rng, -1, True)
                if j.typ != tp or j.n > target - filled: continue
                self._start_job(j, s); j.rem *= float(rng.uniform(0.1, 0.9)); filled += j.n
        self.queue += self.sched[0]; self.sched[0] = []
        return self._build_observation(), {"action_mask": self.action_masks()}

    # ------------------------------------------------------------------ stato
    def free(self, s): return self.cap[s] - sum(j.n for j in self.running if j.sec == s)

    def demand(self): return sum(j.n * j.dem for j in self.running)

    def heads(self):
        h = {}
        for j in self.queue:
            if j.cls not in h: h[j.cls] = j
        return h

    def _future_arrivals(self): return any(self.sched[k] for k in range(self.t, len(self.sched)))

    def _terminal(self): return (not self.queue) and (not self.running) and (not self._future_arrivals())

    def placements(self):
        h = self.heads(); fr = [self.free(0), self.free(1)]; out = []
        for c, j in h.items():
            for s in (0, 1):
                if fr[s] >= j.n: out.append(2 + 2 * c + s)
        return out

    def action_masks(self):
        m = np.zeros(self.n_act, dtype=bool); pl = self.placements()
        for a in pl: m[a] = True
        if self.allow_wait or not pl: m[:len(self.levels)] = True
        return m

    # ------------------------------------------------------------------ dinamica
    def step(self, action):
        a = int(action)
        if a >= len(self.levels):
            c, s = divmod(a - 2, 2); h = self.heads().get(c)
            if h is None or self.free(s) < h.n: raise ValueError("azione non legale")
            self.queue.remove(h); self._start_job(h, s)
            return self._build_observation(), 0.0, False, False, {"action_mask": self.action_masks()}
        L = self.levels[a]; dt_h = self.dt / 3600.0
        nsys = len(self.queue) + sum(1 for j in self.running if not j.bg)
        cost = (nsys * dt_h + self.w * L * dt_h) / self.N
        self.cost_flow_h += nsys * dt_h / self.N; self.cost_en_h += self.w * L * dt_h / self.N
        D = self.demand(); s = 1.0 if D <= 0 else min(1.0, L * PMAX / D); self.last_L, self.last_s = L, s
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
    def ticks_to_free(self, s, k=4):
        out = []
        for j in self.running:
            if j.sec == s: out += [math.ceil(max(j.rem, 0.0) / self.dt - 1e-9)] * j.n
        out = sorted(out) + [0] * max(0, self.free(s))
        out = sorted(out)[:k]
        return out + [0] * (k - len(out))

    def _build_observation(self):
        ob = [min(self.t / 100.0, 1.0), max(self.T_a - self.t, 0) / float(self.T_a), self.free(0) / self.cap[0], self.free(1) / self.cap[1],
              min(self.demand() / PMAX, 1.0), 1.0 if self.last_L >= 1.0 else self.levels[-1] / self.levels[0], self.last_s]
        by = {c: [] for c in range(self.n_cls)}
        for j in self.queue: by[j.cls].append(j)
        for c in range(self.n_cls):
            q = by[c]
            ob += [min(len(q) / 10.0, 1.0), min(q[0].nom / 14400.0, 1.0) if q else 0.0, min(sum(j.nom for j in q) / 28800.0, 1.0)]
        for s in range(self.n_sec): ob += [min(x / 40.0, 1.0) for x in self.ticks_to_free(s)]
        if self.xobs:                                  # osservazione estesa: le prime qk richieste in coda (ordine di arrivo) e il tempo di liberazione di TUTTI i 16 nodi, ordinati per sezione
            for k in range(self.qk):
                if k < len(self.queue):
                    j = self.queue[k]; ob += [j.n / 4.0, float(j.typ), min(j.nom / 14400.0, 1.0), min((self.t - j.arr) / 50.0, 1.0)]
                else: ob += [0.0, 0.0, 0.0, 0.0]
            ob.append(min(max(len(self.queue) - self.qk, 0) / 20.0, 1.0))
            for s in range(self.n_sec): ob += [min(x / 40.0, 1.0) for x in self.ticks_to_free(s, self.cap[s])]
        return np.clip(np.array(ob, dtype=np.float32), 0.0, 1.0)

    def strip_future(self):
        """Copia onesta: i futuri arrivi sono ignorati (il pianificatore non li conosce)."""
        for k in range(self.t, len(self.sched)): self.sched[k] = []
