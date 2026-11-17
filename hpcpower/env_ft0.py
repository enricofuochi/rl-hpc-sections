"""FT0: variante DETERMINISTICA e CORTA di FTEnv (additiva: env_ft.py non e' toccato).

Differenze rispetto a FTEnv:
  * TUTTI i job dell'episodio sono in coda al tick 0: nessun arrivo futuro, nessuna casualita' dopo reset().
    Il seme determina solo la generazione iniziale (job + cluster pre-riempito), pienamente osservabile. La dinamica e' deterministica.
  * n_jobs fisso; con dt=600 s e 24 job l'episodio ha ~51 decisioni (24 piazzamenti + ~27 tick).
  * Costo, fisica, azioni (14), osservazione (xobs) identici a FTEnv. Con qk=n_jobs la coda e' interamente visibile.
Aggiunge la proprieta' n_actions (il wrapper LightZero la usa). FTArrEnv = FTEnv con arrivi (n_actions incluso), per esperimenti futuri.
"""
import numpy as np
from .env_ft import FTEnv


class FT0Env(FTEnv):
    def __init__(self, n_jobs=24, **kw):
        kw.setdefault("T_a", 1)
        super().__init__(**kw)
        self.n_jobs = int(n_jobs)

    @property
    def n_actions(self): return self.n_act

    def reset(self, seed=None, options=None):
        super(FTEnv, self).reset(seed=seed)
        rng = np.random.default_rng(int(seed) if seed is not None else int(self.np_random.integers(2 ** 31)))
        self._uid = 0; self.t = 0; self.queue = []; self.running = []; self.done_ids = 0; self.last_L = 1.0; self.last_s = 1.0
        self.cost_flow_h = 0.0; self.cost_en_h = 0.0; self.ret = 0.0
        self.sched = [[self._job(rng, 0) for _ in range(self.n_jobs)]]
        self.N = max(1, self.n_jobs)
        u = float(rng.uniform(*self.pf)) if self.pf[1] > 0 else 0.0
        for s, tp in ((0, 0), (1, 1)):
            target = int(round(u * self.cap[s])); filled = 0; tries = 0
            while filled < target and tries < 200:
                tries += 1; j = self._job(rng, -1, True)
                if j.typ != tp or j.n > target - filled: continue
                self._start_job(j, s); j.rem *= float(rng.uniform(0.1, 0.9)); filled += j.n
        self.queue += self.sched[0]; self.sched[0] = []
        return self._build_observation(), {"action_mask": self.action_masks()}


class FTArrEnv(FTEnv):
    @property
    def n_actions(self): return self.n_act
