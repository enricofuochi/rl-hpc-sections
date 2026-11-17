"""16 - FIGURA UNICA 'valore e fiducia' (bias del valore, confidenza e entropia della politica). Palette: Rule grigio, PPO arancione, MuZero blu.

Tre pannelli in riga, definizioni dei dati documentate nelle singole funzioni:
  (A) bias della stima del costo residuo lungo l'episodio di ciascun agente (PPO: dalla testa del valore; MuZero: costo-da-qui della regola meno il risparmio predetto), con banda IC95 della media per bin;
  (B) istogramma della probabilita' dell'azione preferita (PPO: politica; MuZero: visite MCTS) sugli stati della traiettoria della regola;
  (C) entropia sulle azioni legali (nats) lungo l'episodio della regola.
Scrive SOLO file nuovi in output/<ESPERIMENTO>/figures/ (valore_e_fiducia.png/.pdf). Le statistiche si stampano a video, per la caption.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
ESPERIMENTO = "mio_test"
BIN_PROGRESSO = 8                  
LARGHEZZA, ALTEZZA = 11.6, 3.5     # pollici (tre pannelli)
# ============================================================
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NOMI = dict(rule="Rule", ppo="PPO", mz="MuZero")
COL = dict(rule="#898781", ppo="#eb6834", mz="#2a78d6")
TESTO, MUTED, GRIGLIA, ASSI, SFONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": ASSI, "axes.facecolor": SFONDO, "figure.facecolor": SFONDO,
                     "savefig.facecolor": SFONDO, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8})


def carica():
    f = percorsi.OUTPUT / ESPERIMENTO / "tracce.pkl"
    if not f.exists(): raise SystemExit(f"Non trovo {f}: prima script 02 per l'esperimento '{ESPERIMENTO}'.")
    d = pickle.load(f.open("rb")); res = d["res"]
    for p in ("rule", "ppo", "mz"):
        if p not in res: raise SystemExit(f"servono le tracce di {NOMI[p]}")
    seeds = sorted(set(res["rule"]) & set(res["ppo"]) & set(res["mz"])); return res, seeds, float(d["meta"].get("resid_scale", 100.0))


def dati_bias(res, seeds, sc):
    """(costo residuo reale, costo residuo predetto, progresso 0-1) per ogni decisione."""
    return {"ppo": np.array([(r["ctg"], r["info"]["ctg_pred"], i / len(res["ppo"][s]["rows"])) for s in seeds for i, r in enumerate(res["ppo"][s]["rows"])]),
            "mz": np.array([(r["ctg"], r["ctg_rule"] - r["info"]["value"] / sc, i / len(res["mz"][s]["rows"])) for s in seeds for i, r in enumerate(res["mz"][s]["rows"])])}


def dati_fiducia(res, seeds):
    """Distribuzioni di PPO (politica) e MuZero (visite) sugli stati della traiettoria della regola."""
    P, M, pr = [], [], []
    for s in seeds:
        rows = res["rule"][s]["rows"]
        for i, r in enumerate(rows):
            if "ppo" in r["q"] and "mz" in r["q"]: P.append(r["q"]["ppo"][1]["probs"]); M.append(r["q"]["mz"][1]["visits"]); pr.append(i / len(rows))
    return np.array(P), np.array(M), np.array(pr)


def principale():
    res, seeds, sc = carica(); B = BIN_PROGRESSO; x = (np.arange(B) + 0.5) / B * 100
    ent = lambda p: -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0), -1)
    fig, ax = plt.subplots(1, 3, figsize=(LARGHEZZA, ALTEZZA), gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=0.34))
    # (A) bias
    bias = dati_bias(res, seeds, sc)
    for p in ("ppo", "mz"):
        d = bias[p]; b = np.minimum(d[:, 2] * B, B - 1e-9).astype(int); e = d[:, 1] - d[:, 0]
        m = np.array([e[b == j].mean() for j in range(B)]); se = np.array([e[b == j].std() / np.sqrt((b == j).sum()) for j in range(B)])
        ax[0].plot(x, m, color=COL[p], lw=2, label=NOMI[p]); ax[0].fill_between(x, m - 1.96 * se, m + 1.96 * se, color=COL[p], alpha=0.18, lw=0)
        print(f"  (A) {NOMI[p]:7s} bias medio {e.mean():+.3f} h, MAE {np.abs(e).mean():.3f} h su {len(d)} stati; primo bin {m[0]:+.3f} h, ultimo bin {m[-1]:+.3f} h")
    ax[0].axhline(0, color=ASSI, lw=1); ax[0].set_xlabel("Episode progress (%)"); ax[0].set_ylabel("Bias of predicted remaining cost (h)\n(positive: overestimates cost)"); ax[0].legend(loc="upper right")
    ax[0].set_title("(A)  Value prediction bias", pad=6)
    # (B) confidenza e (C) entropia
    P, M, pr = dati_fiducia(res, seeds)
    for p, X in (("ppo", P), ("mz", M)): ax[1].hist(X.max(1), bins=np.linspace(0, 1, 21), color=COL[p], alpha=0.6, label=NOMI[p] + (" (policy)" if p == "ppo" else " (MCTS visits)"))
    ax[1].set_xlabel("Probability of the most-preferred action"); ax[1].set_ylabel("Decisions"); ax[1].legend(loc="upper left"); ax[1].set_title("(B)  Top-action probability", pad=6)
    b = np.minimum(pr * B, B - 1e-9).astype(int)
    for p, X in (("ppo", P), ("mz", M)): ax[2].plot(x, [ent(X[b == j]).mean() for j in range(B)], color=COL[p], lw=2, label=NOMI[p])
    ax[2].set_xlabel("Progress along the rule's episode (%)"); ax[2].set_ylabel("Entropy over legal actions (nats)"); ax[2].set_title("(C)  Policy entropy", pad=6)
    print(f"  (B,C) {len(P)} stati della traiettoria della regola: confidenza media (max prob.) PPO {P.max(1).mean():.2f}, MuZero {M.max(1).mean():.2f}; entropia media PPO {ent(P).mean():.2f}, MuZero {ent(M).mean():.2f}")
    d = percorsi.OUTPUT / ESPERIMENTO / "figures"; d.mkdir(exist_ok=True)
    for e_ in ("png", "pdf"): fig.savefig(d / f"valore_e_fiducia.{e_}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  salvata figures/valore_e_fiducia.png e .pdf")


if __name__ == "__main__":
    principale()
