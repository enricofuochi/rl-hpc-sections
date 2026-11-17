"""14 - FIGURE FINALI per la sezione di policy analysis di FT1.

Come si usa: premi play in VS Code. Non serve LightZero ne' giocare nulla: legge le tracce gia' registrate (script 02) in output/<ESPERIMENTO>/tracce.pkl.
Scrive SOLO file nuovi, in output/<ESPERIMENTO>/figures/ (png e pdf). Le figure vecchie in figure/ non vengono toccate.
  gantt_triplo_[id_]seed<S> Diagramma di Gantt dei job per Rule, PPO e MuZero sullo STESSO episodio (stesso seed dell'episodio d'esempio): sezione veloce e efficiente, striscia del livello L sotto.
  cruscotto_energia        (a) termine di energia del costo scomposto in utile / spreco riducibile / spreco al pavimento; (b) spreco cumulato nell'episodio d'esempio; (c) spreco cumulato medio sui seed.
  cruscotto_energia_2p     Come sopra senza il pannello (b) (episodio d'esempio), se il Gantt e l'episodio d'esempio gia' lo mostrano.
Le statistiche (IC95 appaiati, bootstrap B=10000) e i totali dello spreco si stampano a video: servono per le caption, non sono scritti nelle figure.
Palette dei giocatori: la stessa delle curve di addestramento e del Gantt (Rule grigio, PPO arancione, MuZero blu).
"""
# ============================================================ IMPOSTAZIONI RAPIDE
ESPERIMENTO = "mio_test"          # cartella in output/ con tracce.pkl
GIOCATORI = ("rule", "ppo", "mz") # il primo e' il riferimento
ETICHETTE_JOB = True              # Gantt: scrive il numero del job sopra ai rettangoli (stesso numero = stesso job nelle tre colonne); file gantt_triplo_id_seed<S>
FONT_ID = 6.0                     # dimensione del numero del job
SEED_ESEMPIO = 9097               # episodio d'esempio; None = scelta automatica come nello script 13
# ============================================================
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm

NOMI = dict(rule="Rule", ppo="PPO", mz="MuZero")
COL = dict(rule="#898781", ppo="#eb6834", mz="#2a78d6")
TESTO, MUTED, GRIGLIA, ASSI, SFONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff"
G_RID, G_PAV = "#a9a79d", "#d3d1c8"                                                  # spreco riducibile / spreco al pavimento
ALFA_JOB = 0.55                                                                      # trasparenza dei rettangoli dei job (i numeri restano leggibili)
COL_JOB = dict(compute="#d6453d", memory="#1baf7a", background="#d9d8d2")          # compute rosso, memory verde, background grigio
try:
    from hpcpower.env_ft import SECTIONS; CAP = [s[0] for s in SECTIONS]
except Exception: CAP = [6, 10]                                                       # sezione veloce / efficiente (FT1)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": ASSI, "axes.facecolor": SFONDO, "figure.facecolor": SFONDO,
                     "savefig.facecolor": SFONDO, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8})


def boot(x, B=10000, seed=20261109):
    x = np.asarray(x, float); b = np.random.default_rng(seed).integers(0, len(x), (B, len(x))); m = x[b].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def carica():
    f = percorsi.OUTPUT / ESPERIMENTO / "tracce.pkl"
    if not f.exists(): raise SystemExit(f"Non trovo {f}: prima script 02 per l'esperimento '{ESPERIMENTO}'.")
    d = pickle.load(f.open("rb")); res = d["res"]; P = [p for p in GIOCATORI if p in res]
    S = sorted(set.intersection(*[set(res[p]) for p in P])); kw = d["meta"]["env_kw"]; return res, P, S, float(kw["w"]), float(kw["lmin"])


def scegli_seed(res, P, S):
    if SEED_ESEMPIO is not None: return SEED_ESEMPIO
    rif = P[0]; D = {p: np.array([res[p][s]["total"] - res[rif][s]["total"] for s in S]) for p in P[1:]}; m = {p: D[p].mean() for p in P[1:]}
    return S[int(np.argmin(sum(np.abs(D[p] - m[p]) for p in P[1:])))]


def chiudi(fig, nome, d):
    for e in ("png", "pdf"): fig.savefig(d / f"{nome}.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  salvata figures/{nome}.png e .pdf")


# ------------------------------------------------------------------ Gantt
def _cL(L): return cm.Greys(0.30 + 0.70 * (min(1.0, max(0.5, L)) - 0.5) / 0.5)


def _fc_job(j):
    if j["arr"] < 0: return COL_JOB["background"]
    return matplotlib.colors.to_rgba(COL_JOB[("compute", "memory")[j["typ"]]], ALFA_JOB)


def pack(tr, sec):
    cap = CAP[sec]; free = np.zeros(cap); out = []
    for j in sorted([j for j in tr["jobs_bg"] + tr["jobs"] if j["sec"] == sec], key=lambda j: (j["start"], j["id"])):
        idx = [k for k in range(cap) if free[k] <= j["start"]][:j["n"]]
        for k in idx: free[k] = j["end"]; out.append((k, j))
    return cap, out


def gantt(res, P, seed):
    n = len(P); fig, ax = plt.subplots(3, n, figsize=((4.4 if ETICHETTE_JOB else 3.7) * n, 5.4), gridspec_kw=dict(height_ratios=[6, 10, 0.9]), squeeze=False)
    xmax = max((res[p][seed]["ticks"][-1]["t"] + 1) * res[p][seed]["dt"] / 3600.0 for p in P) * 1.02
    for c, p in enumerate(P):
        tr = res[p][seed]; dt = tr["dt"] / 3600.0
        for r, sec in enumerate((0, 1)):
            a = ax[r][c]; cap, pk = pack(tr, sec)
            for k, j in pk: a.add_patch(plt.Rectangle((j["start"] * dt, k + 0.06), (j["end"] - j["start"]) * dt, 0.88, fc=_fc_job(j), ec=SFONDO, lw=0.7))
            if ETICHETTE_JOB:                                                         # numero del job su ogni rettangolo (un job da n nodi ha n rettangoli con lo stesso numero): si segue lo stesso job nelle tre colonne
                for k, j in pk:
                    if j["arr"] >= 0: a.text((j["start"] + j["end"]) / 2 * dt, k + 0.5, str(j["id"]), ha="center", va="center", fontsize=FONT_ID, color=TESTO)
            a.set_xlim(0, xmax); a.set_ylim(0, cap); a.set_yticks([]); a.grid(False); a.tick_params(labelbottom=False)
            a.set_ylabel(f"{'Fast' if sec == 0 else 'Efficient'} section ({cap} nodes)" if c == 0 else "")
        a = ax[2][c]
        for t in tr["ticks"]: a.add_patch(plt.Rectangle((t["t"] * dt, 0), dt, 1, fc=_cL(t["L"]), ec=SFONDO, lw=0.5))
        a.set_xlim(0, xmax); a.set_ylim(0, 1); a.set_yticks([]); a.grid(False); a.set_xlabel("Time (h)")
        if c == 0: a.set_ylabel("L", rotation=0, labelpad=10)
        ax[0][c].set_title(NOMI[p], pad=16)
        ax[0][c].text(0, 1.03, f"cost {tr['total']:.2f} h  (flow {tr['flow']:.2f} + energy {tr['energy']:.2f})", transform=ax[0][c].transAxes, fontsize=8, color=MUTED, va="bottom")
        print(f"  Gantt {NOMI[p]:7s} seed {seed}: costo {tr['total']:.3f} h = flusso {tr['flow']:.3f} + energia {tr['energy']:.3f}; fine {len(tr['ticks'])} tick = {len(tr['ticks']) * dt:.2f} h")
    h = [plt.Rectangle((0, 0), 1, 1, fc=f) for f in (matplotlib.colors.to_rgba(COL_JOB["compute"], ALFA_JOB), matplotlib.colors.to_rgba(COL_JOB["memory"], ALFA_JOB), COL_JOB["background"], _cL(1.0), _cL(0.5))]
    fig.legend(h, ["compute job", "memory job", "background job", "L = 1.0", "L = 0.5"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); return fig


# ------------------------------------------------------------------ energia
def parti_energia(t, w, lmin):
    """(utile, spreco riducibile, spreco al pavimento) in ore, nelle unita' del termine di energia del costo."""
    dth = t["dt"] / 3600.0; N = len(t["jobs"]); Pmax = t["ticks"][0]["deliv"] / t["ticks"][0]["L"]; u = rd = fl = 0.0
    for k in t["ticks"]:
        c = w * dth / N / Pmax; u += c * k["used"]; sp = c * (k["deliv"] - k["used"])
        if abs(k["L"] - lmin) < 1e-9: fl += sp
        else: rd += sp
    return u, rd, fl


def spreco_cum(t):
    dth = t["dt"] / 3600.0; return np.cumsum([(k["deliv"] - k["used"]) / 1000.0 * dth for k in t["ticks"]])


def pannello_barre(ax, V, P):
    x = np.arange(len(P)); bot = np.zeros(len(P))
    for i, (nome, col) in enumerate((("useful energy (player colour)", None), ("reducible waste (L above the minimum)", G_RID), ("floor waste (L at the minimum)", G_PAV))):
        y = np.array([V[p][:, i].mean() for p in P]); cols = [COL[p] if col is None else col for p in P]
        ax.bar(x, y, 0.62, bottom=bot, color=cols, edgecolor=SFONDO, lw=0.8, label=nome)
        for xi, yi, bi in zip(x, y, bot):
            if yi > 0.035: ax.text(xi, bi + yi / 2, f"{yi:.2f}", ha="center", va="center", fontsize=8, color=("white" if col is None else TESTO))
            elif yi > 0: ax.text(xi + 0.34, bi + yi / 2, f"{yi:.2f}", ha="left", va="center", fontsize=8, color=TESTO)      # fetta troppo sottile: etichetta a fianco della barra
        bot += y
    for xi, tot in zip(x, bot): ax.text(xi, tot + 0.015, f"{tot:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([NOMI[p] for p in P]); ax.set_ylabel("Energy term of the cost (h)"); ax.set_ylim(0, bot.max() * 1.17); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7.3); ax.set_title("Energy cost by component")


def pannello_cum_episodio(ax, res, P, seed):
    dth = res[P[0]][seed]["dt"] / 3600.0
    for p in P:
        c = spreco_cum(res[p][seed]); x = np.arange(len(c) + 1) * dth; y = np.append(0.0, c)
        ax.step(x, y, where="post", color=COL[p], lw=2.0, label=f"{NOMI[p]} ({c[-1]:.1f} kWh)"); ax.plot([x[-1]], [y[-1]], "o", color=COL[p], ms=5)
        print(f"  spreco cumulato seed {seed} {NOMI[p]}: {c[-1]:.2f} kWh")
    ax.set_title(f"Cumulative waste, seed {seed}"); ax.set_xlabel("Time (h)"); ax.set_ylabel("Energy (kWh)"); ax.legend(loc="upper left")


def pannello_cum_media(ax, res, P, S):
    dth = res[P[0]][S[0]]["dt"] / 3600.0; K = max(len(res[p][s]["ticks"]) for p in P for s in S)
    for p in P:
        M = np.zeros((len(S), K + 1))
        for i, s in enumerate(S):
            cc = spreco_cum(res[p][s]); M[i, 1:len(cc) + 1] = cc; M[i, len(cc) + 1:] = cc[-1]
        ax.step(np.arange(K + 1) * dth, M.mean(0), where="post", color=COL[p], lw=2.0, label=f"{NOMI[p]} ({M[:, -1].mean():.1f} kWh)")
        print(f"  spreco cumulato medio {NOMI[p]}: {M[:, -1].mean():.2f} kWh")
    ax.set_title("Average cumulative waste"); ax.set_xlabel("Time (h)"); ax.set_ylabel("Energy (kWh)"); ax.legend(loc="upper left")


def principale():
    res, P, S, w, lmin = carica(); seed = scegli_seed(res, P, S)
    print(f"Esperimento '{ESPERIMENTO}': {[NOMI[p] for p in P]}, {len(S)} seed ({S[0]}-{S[-1]}), episodio d'esempio {seed}, w={w:g}, L_min={lmin:g}")
    V = {p: np.array([parti_energia(res[p][s], w, lmin) for s in S]) for p in P}
    for p in P:
        err = np.abs(V[p].sum(1) - np.array([res[p][s]["energy"] for s in S])).max(); assert err < 1e-9, f"le parti non sommano al termine di energia ({p}: {err})"
    print("\nEnergia (h): utile / riducibile / pavimento / totale")
    for p in P: print(f"  {NOMI[p]:7s} {V[p][:, 0].mean():.4f} / {V[p][:, 1].mean():.4f} / {V[p][:, 2].mean():.4f} / {V[p].sum(1).mean():.4f}")
    for p in P[1:]:
        r = lambda i: boot(V[p][:, i] - V[P[0]][:, i]); tot = boot(V[p].sum(1) - V[P[0]].sum(1))
        print(f"  {NOMI[p]} - {NOMI[P[0]]} (appaiato, IC95): totale {tot[0]:+.3f} [{tot[1]:+.3f},{tot[2]:+.3f}]; " + ", ".join(f"{n} {r(i)[0]:+.3f} [{r(i)[1]:+.3f},{r(i)[2]:+.3f}]" for i, n in enumerate(("utile", "riducibile", "pavimento"))))
    d = percorsi.OUTPUT / ESPERIMENTO / "figures"; d.mkdir(exist_ok=True); print(f"\nFigure in {d}/:")
    chiudi(gantt(res, P, seed), f"gantt_triplo_{'id_' if ETICHETTE_JOB else ''}seed{seed}", d)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0])); pannello_barre(axs[0], V, P); pannello_cum_episodio(axs[1], res, P, seed); pannello_cum_media(axs[2], res, P, S)
    fig.tight_layout(); chiudi(fig, "cruscotto_energia", d)
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9)); pannello_barre(axs[0], V, P); pannello_cum_media(axs[1], res, P, S); fig.tight_layout(); chiudi(fig, "cruscotto_energia_2p", d)


if __name__ == "__main__":
    principale()
