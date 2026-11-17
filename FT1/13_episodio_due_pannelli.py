"""13 - EPISODIO IN DUE PANNELLI. Figura grande: stesso episodio (seed 9097), stesso asse del tempo del Gantt.

Colonna sinistra: per Rule, PPO e MuZero, potenza usata (colore del giocatore), spreco riducibile e spreco al pavimento (grigi), linea nera = potenza erogata.
Colonna destra: una riga per giocatore, colonne a colori tenui: potenza RICHIESTA dai job (verde), potenza erogata e non richiesta = spreco (rosso, sopra la richiesta),
                   richiesta che supera l'erogata = throttling (giallo), linea nera a gradini = potenza erogata. Stessi assi in tutti i pannelli, righe allineate per giocatore.
Scrive SOLO file nuovi in output/<ESPERIMENTO>/figures/ (episodio_due_pannelli.png/.pdf). Le statistiche dell'episodio si stampano a video, per la caption.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
ESPERIMENTO = "mio_test"
GIOCATORI = ("rule", "ppo", "mz")
SEED_ESEMPIO = 9097               # lo stesso del Gantt
LARGHEZZA, ALTEZZA = 11.6, 6.9    # pollici (due colonne)
# ============================================================
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

NOMI = dict(rule="Rule", ppo="PPO", mz="MuZero")
COL = dict(rule="#898781", ppo="#eb6834", mz="#2a78d6")
TESTO, MUTED, GRIGLIA, ASSI, SFONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff"
G_RID, G_PAV = "#b9b7ad", "#dcdad1"                                                  # spreco riducibile / al pavimento
VERDE, ROSSO, GIALLO = "#2e9e5b", "#d6453d", "#e8b923"                              # colonne a colori tenui (come episodio_<giocatore>)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": ASSI, "axes.facecolor": SFONDO, "figure.facecolor": SFONDO,
                     "savefig.facecolor": SFONDO, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.8})


def carica():
    f = percorsi.OUTPUT / ESPERIMENTO / "tracce.pkl"
    if not f.exists(): raise SystemExit(f"Non trovo {f}: prima script 02 per l'esperimento '{ESPERIMENTO}'.")
    d = pickle.load(f.open("rb")); res = d["res"]; P = [p for p in GIOCATORI if p in res]; return res, P, float(d["meta"]["env_kw"]["lmin"])


def pannello_sinistro(ax, t, p, lmin, xmax):
    tk = t["ticks"]; dth = t["dt"] / 3600.0; x = np.arange(len(tk) + 1) * dth; used = np.array([k["used"] for k in tk]) / 1000; deliv = np.array([k["deliv"] for k in tk]) / 1000
    fl = np.array([abs(k["L"] - lmin) < 1e-9 for k in tk]); ap = lambda a: np.append(a, a[-1])
    ax.fill_between(x, 0, ap(used), step="post", color=COL[p], alpha=0.9, lw=0)
    ax.fill_between(x, ap(used), ap(np.where(~fl, deliv, used)), step="post", color=G_RID, lw=0)
    ax.fill_between(x, ap(used), ap(np.where(fl, deliv, used)), step="post", color=G_PAV, lw=0)
    ax.step(x, ap(deliv), where="post", color=TESTO, lw=1.0)
    ax.text(0.99, 0.93, f"ends at {len(tk) * dth:.2f} h   energy {t['energy']:.2f} h   total cost {t['total']:.2f} h", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444")


def pannello_destro(ax, t, p, xmax):
    tk = t["ticks"]; dth = t["dt"] / 3600.0; x0 = np.arange(len(tk)) * dth
    D = np.array([k["D"] for k in tk]) / 1000.0; dl = np.array([k["deliv"] for k in tk]) / 1000.0; thr = D > dl + 1e-9; kw = dict(align="edge", edgecolor=SFONDO, lw=0.5)
    ax.bar(x0[~thr], D[~thr], dth, color=VERDE, alpha=0.45, **kw)
    ax.bar(x0[~thr], (dl - D)[~thr], dth, bottom=D[~thr], color=ROSSO, alpha=0.40, **kw)
    ax.bar(x0[thr], D[thr], dth, color=GIALLO, alpha=0.55, **kw)
    xs = np.append(x0, x0[-1] + dth); ax.step(xs, np.append(dl, dl[-1]), where="post", color=TESTO, lw=1.2)
    spreco = float(np.sum(np.maximum(dl - D, 0)) * dth)
    ax.text(0.99, 0.93, f"wasted {spreco:.1f} kWh   throttled ticks {int(thr.sum())}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444")
    print(f"  {NOMI[p]:7s} fine {len(tk) * dth:.2f} h, costo {t['total']:.3f} h, spreco {spreco:.1f} kWh, tick strozzati {int(thr.sum())}/{len(tk)}")


def principale():
    res, P, lmin = carica(); seed = SEED_ESEMPIO; dth = res[P[0]][seed]["dt"] / 3600.0
    xmax = max(len(res[p][seed]["ticks"]) for p in P) * dth * 1.02; ymax = max(12.6, max(max(k["D"] for k in res[p][seed]["ticks"]) for p in P) / 1000.0 * 1.12)
    print(f"Episodio d'esempio: seed {seed}, asse tempo 0-{xmax:.3f} h (come il Gantt)")
    fig, axs = plt.subplots(len(P), 2, figsize=(LARGHEZZA, ALTEZZA), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.22))
    for r, p in enumerate(P):
        pannello_sinistro(axs[r][0], res[p][seed], p, lmin, xmax); pannello_destro(axs[r][1], res[p][seed], p, xmax)
        axs[r][0].set_ylabel(f"{NOMI[p]}\nkW"); axs[r][0].set_xlim(0, xmax); axs[r][0].set_ylim(0, ymax); axs[r][1].tick_params(labelleft=False)
    for ax in axs[-1]: ax.set_xlabel("Time (h)")
    hl = [Patch(color="#555555", alpha=0.9), Patch(color=G_RID), Patch(color=G_PAV), Line2D([0], [0], color=TESTO, lw=1.0)]
    axs[0][0].legend(hl, ["power used (player colour)", "reducible waste", "floor waste", "delivered power"], loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    hr = [Patch(color=VERDE, alpha=0.45), Patch(color=ROSSO, alpha=0.40), Patch(color=GIALLO, alpha=0.55), Line2D([0], [0], color=TESTO, lw=1.2)]
    axs[0][1].legend(hr, ["power requested", "wasted (delivered above requested)", "requested above delivered (throttle)", "delivered power"], loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    axs[0][0].set_title("(A)  Power used and wasted", pad=34); axs[0][1].set_title("(B)  Power requested, wasted and throttled", pad=34)
    d = percorsi.OUTPUT / ESPERIMENTO / "figures"; d.mkdir(exist_ok=True)
    for e in ("png", "pdf"): fig.savefig(d / f"episodio_due_pannelli.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  salvata figures/episodio_due_pannelli.png e .pdf")


if __name__ == "__main__":
    principale()
