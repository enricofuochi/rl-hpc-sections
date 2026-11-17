"""10 - LA RICERCA AIUTAVA ALL'INIZIO? Stesso confronto dello script 09 (sola rete contro ricerca) ma su checkpoint INTERMEDI della run di addestramento.

Come si usa: apri questo file in VS Code e premi play (serve un ambiente Python con torch + LightZero).
Idea: se la ricerca e' l'insegnante (le visite di MCTS sono il bersaglio della rete di politica), all'inizio la rete e' scarsa e la ricerca la corregge molto; man mano che l'addestramento procede la rete
"assorbe" la ricerca e la differenza sparisce. Qui si misura: per ogni checkpoint, costo con la sola rete e con SIMULAZIONI simulazioni, la differenza appaiata (con IC95) e la quota di decisioni in cui la ricerca
cambia la scelta della rete. Se la quota di decisioni cambiate e il beneficio calano con le iterazioni, la ricerca e' stata assorbita; se restano uguali, la ricerca non ha mai fatto la differenza.
Usa i seed di VALIDAZIONE (1000-1029) per non toccare il test (9000-9099, gia' usato per i risultati finali): e' un'analisi descrittiva, non serve a scegliere checkpoint.
Cosa produce: output/ricerca_per_iterazione_<RUN>.csv (riprendibile: se lo rilanci aggiunge solo cio' che manca) e output/figure_ricerca_<RUN>.png/pdf (tre pannelli: costo dei due giocatori, differenza con IC95, decisioni cambiate).
Se il CSV esiste gia' e contiene tutti i checkpoint di ITERAZIONI, non si gioca nulla e si rifanno solo tabella e figura (anche senza i checkpoint).
La ricerca C++ non e' riproducibile: REPLICHE > 1 serve a mediare il rumore. Tempo: la parte lenta e' la ricerca (episodi x checkpoint x repliche); la sola rete e' quasi istantanea.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
RUN = "ft1_lungo_s0"                           # cartella in FT1/runs/ con i checkpoint
ITERAZIONI = (0, 1000, 2000, 3000, 5000, 10000, 25000, 40000, 56880)   # checkpoint mostrati in tabella e figura (devono esistere nei dati o essere in GIOCA)
GIOCA = None                                           # None = gioca solo cio' che manca nel CSV (tra i checkpoint di ITERAZIONI) | tupla = solo questi checkpoint. (0 = rete non addestrata)
SIMULAZIONI = 100                              # simulazioni del confronto. 100 = come in addestramento; 25 = piu' veloce
REPLICHE = 2                                   # repliche della ricerca per seed (la sola rete e' deterministica: 1)
SEED_DA, SEED_A = 1000, 1060                   # validazione (1000-1029) piu' altri 30 seed mai usati. Per ogni checkpoint si usano i seed che hanno tutti i dati. (9000-9100 = test: sconsigliato, e' il test finale)
SALVA = True
# ============================================================
import sys, time, csv, glob, importlib.util, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import ft1_config as C, ft1_giocatori as G
from ft1lib.factory_ft1 import make_env
from ft1lib.policies_ft1 import FT1_POLICIES

_spec = importlib.util.spec_from_file_location("ricerca09", pathlib.Path(__file__).resolve().parent / "09_ricerca_vs_rete.py"); R9 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R9)    # riusa agente(), prior(), boot()
BLU, ARANCIO, GRIGIO, VERDE, VIOLA = R9.BLU, R9.ARANCIO, R9.GRIGIO, R9.VERDE, R9.VIOLA


def checkpoint(it):
    c = sorted(glob.glob(str(percorsi.RUNS / RUN / "data_mz" / "*" / "ckpt" / f"iteration_{it}.pth.tar")))
    if not c: raise SystemExit(f"Non trovo il checkpoint {it} in {percorsi.RUNS / RUN}")
    return c[0]


def gioca(ag, kw, s, sims):
    """Un episodio; restituisce (costo in ore, decisioni, decisioni in cui la ricerca ha scelto un'azione diversa da quella della sola rete)."""
    import torch
    G.seed_all(s); e = make_env(dict(kw)); e.reset(seed=s); cost = 0.0; n = 0; over = 0
    while True:
        ap = R9.prior(ag, e)[0]
        if sims == 0: a = ap
        else:
            obs = torch.tensor(e._build_observation()[None], dtype=torch.float32); mask = e.action_masks().astype(np.int8)
            with torch.no_grad(): a = int(ag.policy.eval_mode.forward(obs, [mask], to_play=[-1], ready_env_id=np.array([0]), timestep=[0])[0]["action"])
            over += int(a != ap)
        _, r, te, tr, _ = e.step(int(a)); cost -= r / 3600.0; n += 1
        if te or tr: return cost, n, over
        assert n < 1000


def costo_regola(kw, s, regola):
    pol = FT1_POLICIES[regola]; G.seed_all(s); e = make_env(dict(kw)); e.reset(seed=s); cost = 0.0
    while True:
        _, r, te, tr, _ = e.step(int(pol(e))); cost -= r / 3600.0
        if te or tr: return cost


def principale():
    cfg = C.carica("fissa"); kw = C.env_kw(cfg); seeds = list(range(SEED_DA, SEED_A)); f = percorsi.OUTPUT / f"ricerca_per_iterazione_{RUN}.csv"
    gioca_it = tuple(ITERAZIONI if GIOCA is None else GIOCA)
    print(f"Run {RUN}: da giocare i checkpoint {list(gioca_it)}; {len(seeds)} seed ({SEED_DA}-{SEED_A - 1}); ricerca a {SIMULAZIONI} simulazioni x {REPLICHE} repliche")
    rule = {s: costo_regola(kw, s, cfg["REGOLA"]) for s in seeds}; print(f"Regola {cfg['REGOLA']}: costo medio {np.mean(list(rule.values())):.3f} h sui seed scelti")
    dati = {}                                                                                   # (it, sims, replica, seed) -> (costo, n, over)
    if f.exists():
        for r in csv.DictReader(f.open()): dati[(int(r["it"]), int(r["sims"]), int(r["replica"]), int(r["seed"]))] = (float(r["cost"]), int(r["n"]), int(r["over"]))
    for it in gioca_it:
        for sims, reps in ((0, 1), (SIMULAZIONI, REPLICHE)):
            todo = [(rep, s) for rep in range(reps) for s in seeds if (it, sims, rep, s) not in dati]
            if not todo: continue
            ag = R9.agente(checkpoint(it), sims); t0 = time.time(); print(f"\nit {it}, {'sola rete' if sims == 0 else f'{sims} simulazioni'}: {len(todo)} episodi", flush=True)
            for i, (rep, s) in enumerate(todo):
                c, n, o = gioca(ag, kw, s, sims); dati[(it, sims, rep, s)] = (c, n, o)
                with f.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if f.stat().st_size == 0: w.writerow(["it", "sims", "replica", "seed", "cost", "n", "over"])
                    w.writerow([it, sims, rep, s, f"{c:.6f}", n, o])
                if i in (0, 4) or (i + 1) % 20 == 0: print(f"  {i + 1}/{len(todo)}  ({(time.time() - t0) / (i + 1):.1f} s per episodio)", flush=True)
    righe = []
    print(f"\n{'iterazione':>10} {'rete vs regola (h)':>26} {'ricerca vs regola (h)':>26} {'rete - ricerca (h)':>26} {'decisioni cambiate':>20}")
    for it in ITERAZIONI:
        ss = [s for s in seeds if (it, 0, 0, s) in dati and all((it, SIMULAZIONI, r, s) in dati for r in range(REPLICHE))]
        if not ss: print(f"{it:>10}  (nessun dato: aggiungilo a GIOCA)"); continue
        net = np.array([dati[(it, 0, 0, s)][0] for s in ss]); se = np.array([np.mean([dati[(it, SIMULAZIONI, r, s)][0] for r in range(REPLICHE)]) for s in ss]); rl = np.array([rule[s] for s in ss])
        over = sum(dati[(it, SIMULAZIONI, r, s)][2] for r in range(REPLICHE) for s in ss) / max(1, sum(dati[(it, SIMULAZIONI, r, s)][1] for r in range(REPLICHE) for s in ss))
        a = R9.boot(net - rl); b = R9.boot(se - rl); d = R9.boot(net - se); righe.append(dict(it=it, net=a, search=b, diff=d, over=over, n=len(ss)))
        print(f"{it:>10} {a[0]:+8.3f} [{a[1]:+.3f},{a[2]:+.3f}] {b[0]:+8.3f} [{b[1]:+.3f},{b[2]:+.3f}] {d[0]:+8.3f} [{d[1]:+.3f},{d[2]:+.3f}] {100 * over:19.1f}%  (n={len(ss)})")
    print("\nLettura: 'rete - ricerca' > 0 con IC che esclude 0 = la ricerca migliora il costo a quel checkpoint. Se la quota di decisioni cambiate e il beneficio calano con le iterazioni, la rete ha assorbito la ricerca.")
    if SALVA: figura(righe)


def figura(righe):
    """Tre pannelli: A costo dei due giocatori, B differenza (cosa aggiunge la ricerca, con IC95), C quota di decisioni cambiate.
    Scala e zona 'la ricerca aiuta' si calcolano dai dati; i punti fuori scala (tipicamente l'iterazione 0) si mostrano con un triangolo e il valore scritto."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK, MUT, SAND = "#2b2b2b", "#6b6b6b", "#f3e3c8"
    plt.rcParams.update({"font.size": 9, "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff", "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": R9.GRIGLIA, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "#ffffff", "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT})
    x = [r["it"] for r in righe]; resto = [r for r in righe if r["it"] > 0] or righe
    def limiti(chiavi):                                                                         # scala scelta sui checkpoint > 0 (+ lo zero), cosi' l'iterazione 0 non schiaccia il resto
        v = [0.0] + [r[k][i] for r in resto for k in chiavi for i in (1, 2)]; a, b = min(v), max(v); m = 0.10 * (b - a); return a - m, b + m
    keep, ultimo = set(), None                                                                  # etichette dell'asse x: si salta quella troppo vicina alla successiva
    for it in reversed(x):
        if ultimo is None or ultimo / max(it, 1) >= 1.45: keep.add(it); ultimo = it
    util = [r["it"] for r in righe if r["diff"][1] > 0]                                         # checkpoint in cui l'IC95 della differenza esclude 0 a favore della ricerca
    if util:
        dopo = [i for i in x if i > max(util)]; banda = (min(util) * 0.6 if min(util) > 0 else -1, dopo[0] if dopo else max(util) * 1.3)
    def asse(a):
        a.set_xscale("symlog", linthresh=1000); a.set_xticks(x); a.set_xticklabels([(f"{i // 1000}k" if i >= 1000 else str(i)) if i in keep else "" for i in x]); a.minorticks_off(); a.set_xlabel("Learner iteration (log scale above 1k)")
        if util: a.axvspan(banda[0], banda[1], color=SAND, alpha=.55, lw=0, zorder=0)
    def serie(a, chiave, col, mk, lab, off, lim):
        m = np.array([r[chiave][0] for r in righe]); lo = np.array([r[chiave][1] for r in righe]); hi = np.array([r[chiave][2] for r in righe]); xx = np.array([i + off * (0.06 * i if i else 45) for i in x], float)                    # piccolo scarto orizzontale tra i due giocatori (anche a iterazione 0)
        k = 1 if x[0] == 0 else 0                                                               # l'iterazione 0 non si collega al resto: il salto non e' una curva
        a.errorbar(xx[k:], m[k:], yerr=[(m - lo)[k:], (hi - m)[k:]], fmt=mk + "-", color=col, lw=1.7, ms=4.5, capsize=2.5, label=lab)
        if k:
            if lim[0] <= m[0] <= lim[1]: a.errorbar([xx[0]], [m[0]], yerr=[[m[0] - lo[0]], [hi[0] - m[0]]], fmt=mk, color=col, ms=4.5, capsize=2.5)
            else:
                alto = m[0] > lim[1]; a.plot([xx[0]], [lim[1] - 0.04 * (lim[1] - lim[0]) if alto else lim[0] + 0.04 * (lim[1] - lim[0])], "^" if alto else "v", color=col, ms=7, clip_on=False)
                a.text(120, (lim[1] - 0.03 * (lim[1] - lim[0])) if alto else (lim[0] + 0.12 * (lim[1] - lim[0])), f"it 0: {m[0]:+.2f} h\n(off scale)", fontsize=8, color=col, va="top")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.8]))
    a = ax[0]; lim = limiti(("net", "search")); a.axhline(0, color=MUT, lw=1); a.set_ylim(*lim)
    serie(a, "net", R9.ARANCIO, "s", "network only", -1, lim); serie(a, "search", R9.BLU, "o", f"search ({SIMULAZIONI} sims)", 1, lim)
    asse(a); a.set_ylabel("Cost minus rule cost (h)   (lower = better)"); a.set_title("A. Cost of the two players", loc="left", fontweight="bold"); a.legend(loc="lower left")
    a = ax[1]; lim = limiti(("diff",)); a.axhline(0, color=MUT, lw=1); a.set_ylim(*lim); serie(a, "diff", INK, "o", "", 0, lim)
    asse(a); a.set_ylabel("Network cost minus search cost (h)"); a.set_title("B. What the search adds", loc="left", fontweight="bold")
    if util: a.text(np.sqrt(max(banda[0], 1) * banda[1]), lim[1] - 0.06 * (lim[1] - lim[0]), "search helps", ha="center", fontsize=8.5, color="#9a5d12")
    tardi = [i for i in x if util and i > max(util)]
    if tardi: a.text(np.sqrt(tardi[0] * tardi[-1]), 0.48 * lim[1], f"no measurable benefit\nafter ~{tardi[0] // 1000}k iterations", ha="center", fontsize=8.5, color=MUT)
    a = ax[2]; ov = [100 * r["over"] for r in righe]; a.plot(x, ov, "o-", color=R9.VIOLA, lw=2, ms=4.5)
    for i, o in zip(x, ov): a.text(i, o + 0.045 * max(ov), f"{o:.0f}", ha="center", fontsize=7.5, color=MUT)
    asse(a); a.set_ylim(0, 1.18 * max(ov)); a.set_ylabel("Decisions changed by search (%)"); a.set_title("C. How often search overrides the network", loc="left", fontweight="bold")
    gruppi = []
    for r in righe:
        if gruppi and gruppi[-1][0] == r["n"]: gruppi[-1][2] = r["it"]
        else: gruppi.append([r["n"], r["it"], r["it"]])
    print("[per la didascalia] " + f"Paired differences over seeds {SEED_DA}-{SEED_A - 1}; seeds per checkpoint: " + ", ".join(f"{n} (it {a_}-{b_})" for n, a_, b_ in gruppi) + f"; {REPLICHE} search replicas; 95% bootstrap CI. Shaded band: checkpoints where the search improves the cost.")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(percorsi.OUTPUT / f"figure_ricerca_{RUN}.{e}", dpi=200, bbox_inches="tight")
    print(f"Figura salvata in {percorsi.OUTPUT}/figure_ricerca_{RUN}.png e .pdf")


if __name__ == "__main__":
    principale()
