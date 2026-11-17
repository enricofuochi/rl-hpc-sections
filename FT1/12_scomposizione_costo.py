"""12 - SCOMPOSIZIONE DEL COSTO: da dove viene il guadagno degli agenti rispetto alla regola.

Come si usa: apri questo file in VS Code e premi play. Non serve LightZero ne' giocare nulla: legge le tracce gia' registrate (script 02) in output/<ESPERIMENTO>/tracce.pkl.
Costo = flusso + energia (ore). Il FLUSSO e' il tempo medio di permanenza dei job nel sistema: poiche' tutti i job sono in coda a t = 0, coincide con il tempo medio di completamento.
L'ENERGIA e' w * (ore a piena erogazione equivalenti) / N: la potenza erogata L * P_max si paga tutta a ogni tick, fino all'ultimo job.
Cosa produce (differenze SEMPRE appaiate sullo stesso seed rispetto alla regola, con IC95 per bootstrap sui seed):
  [TABELLA A]  costo, flusso, energia, makespan, L medio, tick, quota di potenza erogata e non usata, throttling medio; assoluti e differenze.
               Se L medio e' uguale ma l'energia scende, il risparmio viene da episodi piu' corti, non da una potenza piu' bassa.
  [TABELLA B]  contributo di ogni classe di job (compute/memory x 1, 2, 4 nodi) e di ogni terzile di durata al guadagno di flusso: la somma delle classi e' la differenza di flusso totale.
               In piu' attesa, tempo di esecuzione e quota di piazzamenti sulla sezione veloce per tipo di job.
Scrive in output/<ESPERIMENTO>/: scomposizione_costo.csv, contributi_classe.csv e figures/ (un pannello per file: scomposizione_flusso_energia, contributi_classe, piazzamento_sezioni; piu' la figura d'insieme).
Giocatori letti: rule, ppo, mz (quelli assenti dalle tracce si saltano). I seed sono quelli che hanno tutti i giocatori.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
ESPERIMENTO = "mio_test"          # cartella in output/ con tracce.pkl (riferimento = solo regola/PPO/PLAN; mio_test = anche MuZero)
GIOCATORI = ("rule", "ppo", "mz") # il primo e' il riferimento delle differenze
SALVA = True
# ============================================================
import sys, pathlib, pickle, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NOMI = dict(rule="Rule", ppo="PPO", mz="MuZero", plan1="PLAN 1 step", plan2="PLAN 2 steps")
COL = dict(rule="#898781", ppo="#eb6834", mz="#2a78d6", plan1="#6250d6", plan2="#6250d6"); TESTO, GRIGLIA, SFONDO = "#0b0b0b", "#e1e0d9", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7", "axes.facecolor": SFONDO, "figure.facecolor": SFONDO, "savefig.facecolor": SFONDO,
                     "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8})
CLASSI = [(t, n) for t in (0, 1) for n in (4, 2, 1)]                         # (tipo 0 = compute, 1 = memory; nodi)
NCL = lambda c: f"{'compute' if c[0] == 0 else 'memory'} {c[1]} nodi"


def boot(x, B=10000, seed=20261109):
    x = np.asarray(x, float); b = np.random.default_rng(seed).integers(0, len(x), (B, len(x))); m = x[b].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def carica():
    f = percorsi.OUTPUT / ESPERIMENTO / "tracce.pkl"
    if not f.exists(): raise SystemExit(f"Non trovo {f}: prima script 02 (tracce complete) per l'esperimento '{ESPERIMENTO}'.")
    res = pickle.load(f.open("rb"))["res"]; P = [p for p in GIOCATORI if p in res]
    if len(P) < 2: raise SystemExit(f"Servono almeno due giocatori in {f} (trovati: {list(res)})")
    S = sorted(set.intersection(*[set(res[p]) for p in P])); return res, P, S


METRICHE = {                                                                   # nome -> (funzione sulla traccia, formato)
    "costo totale (h)": lambda t: t["total"], "flusso (h)": lambda t: t["flow"], "energia (h)": lambda t: t["energy"], "makespan (h)": lambda t: t["makespan_h"],
    "L medio": lambda t: np.mean([k["L"] for k in t["ticks"]]), "tick": lambda t: len(t["ticks"]),
    "potenza erogata non usata (quota)": lambda t: sum(k["deliv"] - k["used"] for k in t["ticks"]) / sum(k["deliv"] for k in t["ticks"]),
    "throttling medio (1 - s)": lambda t: np.mean([1 - k["s"] for k in t["ticks"]]),
}


def tabella_A(res, P, S):
    V = {p: {m: np.array([f(res[p][s]) for s in S]) for m, f in METRICHE.items()} for p in P}; rif = P[0]; righe = []
    print(f"\nTABELLA A  -  {len(S)} seed ({S[0]}-{S[-1]}), esperimento '{ESPERIMENTO}'. Differenze appaiate rispetto a {NOMI[rif]} con IC95.")
    print(f"{'':36s}" + "".join(f"{NOMI[p]:>10s}" for p in P) + "   |   " + "   ".join(f"{NOMI[p] + ' - ' + NOMI[rif]:>30s}" for p in P[1:]))
    for m in METRICHE:
        ass = "".join(f"{V[p][m].mean():10.4f}" for p in P); dif = []
        for p in P[1:]:
            b = boot(V[p][m] - V[rif][m]); dif.append(f"{b[0]:+8.4f} [{b[1]:+.4f},{b[2]:+.4f}]"); righe.append([m, NOMI[p], f"{V[p][m].mean():.5f}", f"{V[rif][m].mean():.5f}", f"{b[0]:.5f}", f"{b[1]:.5f}", f"{b[2]:.5f}"])
        print(f"{m:36s}{ass}   |   " + "   ".join(dif))
    print("Lettura: se 'L medio' non cambia ma 'energia' scende, il risparmio viene da un makespan piu' corto (meno tick pagati), non da una potenza piu' bassa.")
    return V, righe


def classe_job(j): return (j["typ"], j["n"])


def tabella_B(res, P, S):
    rif = P[0]; J = {p: {s: {j["id"]: j for j in res[p][s]["jobs"]} for s in S} for p in P}; DT = res[rif][S[0]]["dt"] / 3600.0; N = len(J[rif][S[0]])
    for s in S:                                                                # i job devono essere gli stessi per tutti i giocatori (stesso seed)
        base = {i: (j["n"], j["typ"], round(j["nom"], 3)) for i, j in J[rif][s].items()}
        for p in P[1:]: assert base == {i: (j["n"], j["typ"], round(j["nom"], 3)) for i, j in J[p][s].items()}, f"job diversi al seed {s}"
    noms = np.array([j["nom"] for s in S for j in J[rif][s].values()]); q = np.percentile(noms, [100 / 3, 200 / 3]); terz = lambda j: 0 if j["nom"] < q[0] else (1 if j["nom"] < q[1] else 2)
    def contrib(p, filtro): return np.array([sum((J[p][s][i]["end"] - J[rif][s][i]["end"]) * DT for i, j in J[rif][s].items() if filtro(j)) / N for s in S])
    righe = []; out = {}
    print("\nTABELLA B  -  contributo di ogni classe di job al guadagno di FLUSSO (h per episodio; la somma delle classi e' la differenza di flusso totale), IC95 appaiato")
    print(f"{'classe':18s}{'job/ep':>8s}" + "".join(f"{NOMI[p]:>30s}" for p in P[1:]))
    for c in CLASSI:
        n = np.mean([sum(1 for j in J[rif][s].values() if classe_job(j) == c) for s in S]); cells = []
        for p in P[1:]:
            v = contrib(p, lambda j: classe_job(j) == c); b = boot(v); out[(p, c)] = b; cells.append(f"{b[0]:+8.4f} [{b[1]:+.4f},{b[2]:+.4f}]"); righe.append(["classe", NCL(c), NOMI[p], f"{n:.2f}", f"{b[0]:.5f}", f"{b[1]:.5f}", f"{b[2]:.5f}"])
        print(f"{NCL(c):18s}{n:8.1f}" + "".join(f"{x:>30s}" for x in cells))
    print(f"{'TOTALE':18s}{'':8s}" + "".join(f"{np.mean([sum((J[p][s][i]['end'] - J[rif][s][i]['end']) * DT for i in J[rif][s]) / N for s in S]):+30.4f}" for p in P[1:]))
    print(f"\nPer terzile di durata nominale (soglie {q.round(0)} s):")
    for k, nome in enumerate(("corti", "medi", "lunghi")):
        cells = []
        for p in P[1:]:
            v = contrib(p, lambda j: terz(j) == k); b = boot(v); cells.append(f"{b[0]:+8.4f} [{b[1]:+.4f},{b[2]:+.4f}]"); righe.append(["terzile", nome, NOMI[p], "", f"{b[0]:.5f}", f"{b[1]:.5f}", f"{b[2]:.5f}"])
        print(f"{nome:18s}{'':8s}" + "".join(f"{x:>30s}" for x in cells))
    print("\nAttesa in coda (start) / esecuzione (end - start), in ore, e quota di piazzamenti sulla sezione veloce, per tipo di job:")
    sez = {}
    for typ, nome in ((0, "compute"), (1, "memory")):
        for p in P:
            js = [j for s in S for j in J[p][s].values() if j["typ"] == typ]
            w = np.mean([j["start"] * DT for j in js]); r = np.mean([(j["end"] - j["start"]) * DT for j in js]); f = 100 * np.mean([j["sec"] == 0 for j in js]); sez[(p, typ)] = f
            print(f"  {nome:8s} {NOMI[p]:7s} attesa {w:5.2f}  esecuzione {r:5.2f}  sezione veloce {f:5.1f}%"); righe.append(["tipo", nome, NOMI[p], "", f"{w:.4f}", f"{r:.4f}", f"{f:.2f}"])
    return out, sez, righe


def figure(V, P, S, out, sez):
    rif = P[0]; A = P[1:]; w = 0.8 / len(A); figs = {}
    def pannello_1(ax, tit):
        ms = ["flusso (h)", "energia (h)", "costo totale (h)"]; lab = ["flow", "energy", "total cost"]
        for k, p in enumerate(A):
            b = [boot(V[p][m] - V[rif][m]) for m in ms]; x = np.arange(3) + (k - (len(A) - 1) / 2) * w; y = np.array([v[0] for v in b])
            ax.bar(x, y, w * 0.92, color=COL[p], label=NOMI[p], yerr=[y - [v[1] for v in b], [v[2] for v in b] - y], capsize=3, error_kw=dict(lw=1, color=TESTO))
        ax.axhline(0, color="#898781", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylabel(f"Difference vs {NOMI[rif]} (h, paired)"); ax.set_title(f"{tit}Where the cost saving comes from"); ax.legend(loc="lower left")
        ml = [boot(V[p]["L medio"] - V[rif]["L medio"])[0] for p in A]; print("  [per la caption] mean L change vs " + NOMI[rif] + ": " + ", ".join(f"{NOMI[p]} {x:+.3f}" for p, x in zip(A, ml)) + "  (no lower power: shorter episodes)")
    def pannello_2(ax, tit):
        for k, p in enumerate(A):
            b = [out[(p, c)] for c in CLASSI]; x = np.arange(len(CLASSI)) + (k - (len(A) - 1) / 2) * w; y = np.array([v[0] for v in b])
            ax.bar(x, y, w * 0.92, color=COL[p], label=NOMI[p], yerr=[y - [v[1] for v in b], [v[2] for v in b] - y], capsize=3, error_kw=dict(lw=1, color=TESTO))
        ax.axhline(0, color="#898781", lw=1); ax.set_xticks(range(len(CLASSI))); ax.set_xticklabels([NCL(c).replace(" nodi", "\n nodes").replace("compute", "compute").replace("memory", "memory") for c in CLASSI], fontsize=8)
        ax.set_ylabel(f"Contribution to flow change vs {NOMI[rif]} (h)"); ax.set_title(f"{tit}Which jobs gain (the classes sum to the flow change)"); ax.legend(loc="lower right")
    def pannello_3(ax, tit):
        for k, p in enumerate(P):
            x = np.arange(2) + (k - (len(P) - 1) / 2) * (0.8 / len(P)); y = [sez[(p, 0)], sez[(p, 1)]]; ax.bar(x, y, 0.8 / len(P) * 0.92, color=COL[p], label=NOMI[p])
            for xi, yi in zip(x, y): ax.text(xi, yi + 1.5, f"{yi:.0f}%", ha="center", fontsize=7.5, color=TESTO)
        ax.set_xticks(range(2)); ax.set_xticklabels(["compute jobs", "memory jobs"]); ax.set_ylim(0, 105); ax.set_ylabel("Placed on the fast section (%)"); ax.set_title(f"{tit}Job-to-section matching"); ax.legend(loc="upper right")
    pann = {"scomposizione_flusso_energia": pannello_1, "contributi_classe": pannello_2, "piazzamento_sezioni": pannello_3}
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for ax, (nome, f), L in zip(axs, pann.items(), "ABC"): f(ax, f"{L}. ")
    fig.tight_layout(); figs["insieme"] = fig
    for nome, f in pann.items():
        fg, ax = plt.subplots(figsize=(6.4, 4.2)); f(ax, ""); fg.tight_layout(); figs[nome] = fg
    return figs


def principale():
    res, P, S = carica(); print(f"Esperimento '{ESPERIMENTO}': giocatori {[NOMI[p] for p in P]}, {len(S)} seed ({S[0]}-{S[-1]})")
    V, rA = tabella_A(res, P, S); out, sez, rB = tabella_B(res, P, S)
    if not SALVA: return
    d = percorsi.OUTPUT / ESPERIMENTO; (d / "figures").mkdir(exist_ok=True)
    with (d / "scomposizione_costo.csv").open("w", newline="") as fh: w = csv.writer(fh); w.writerow(["metrica", "giocatore", "media_giocatore", "media_riferimento", "differenza", "ic95_basso", "ic95_alto"]); w.writerows(rA)
    with (d / "contributi_classe.csv").open("w", newline="") as fh: w = csv.writer(fh); w.writerow(["gruppo", "nome", "giocatore", "job_per_episodio", "valore", "ic95_basso", "ic95_alto"]); w.writerows(rB)
    figs = figure(V, P, S, out, sez)
    for nome, fg in figs.items():
        base = d / "figures" / (f"scomposizione_costo_{nome}" if nome == "insieme" else nome)
        for e in ("png", "pdf"): fg.savefig(f"{base}.{e}", dpi=200, bbox_inches="tight")
        plt.close(fg)
    print(f"\nSalvato in {d}: scomposizione_costo.csv, contributi_classe.csv, figures/ (insieme + 3 pannelli)")


if __name__ == "__main__":
    principale()
