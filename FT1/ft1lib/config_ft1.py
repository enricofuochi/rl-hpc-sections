"""Parametri di FT1 in un posto solo (usati da ambiente, regole, MuZero, PPO, valutazioni)."""

# Ambiente: identico a FT0 (24 job a t=0, tick 600 s, cluster pre-riempito 70-80%) salvo il controllo della potenza (vedi env_ft1.py).
KW = dict(env="ft1", n_jobs=24, dt=600.0, w=5.7, pf=(0.70, 0.80), dur_med=1800.0, sigma=1.0, max_ticks=400,
          delta=0.1, lmin=0.5, xobs=True, qk=24)

# Regola di riferimento (anche per il reward di scarto di MuZero): scelta sui seed di calibrazione 900-929 fra le 15 regole (3 ordini x 5 livelli): la migliore, anche in validazione (vedi 01_baseline.py).
REGOLA = "ft1_sjf_trk05"

SEED_CAL = tuple(range(900, 930))        # calibrazione di regole e parametri (come per FT0)
SEED_VAL = tuple(range(1000, 1030))      # validazione (descrittiva)
SEED_TEST = tuple(range(9000, 9100))     # test finale: seed FRESCHI, mai usati da nessun altro esperimento (FT0 usa 7000-7099)
SEED_TRAIN_MIN = 10_000                  # i seed di raccolta/addestramento sono casuali >= 10000
