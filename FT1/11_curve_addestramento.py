"""11 - CURVE DI ADDESTRAMENTO di una run MuZero: sei pannelli, ognuno salvabile anche da solo (per metterli in punti diversi della tesi).

Come si usa: apri questo file in VS Code e premi play. Non serve LightZero ne' giocare nulla: legge i file TensorBoard della run e due CSV gia' prodotti.
Pannelli (cartella output/curve_<RUN>/, un file per pannello, png e pdf, piu' la figura d'insieme output/curve_addestramento_<RUN>.png/pdf):
  A  costo dei checkpoint su seed di sviluppo (1000-1029) contro la regola, con IC95 della differenza appaiata        -> costo_checkpoint
  B  ritorno in addestramento (raccolta con esplorazione) e valutazioni interne, convertiti in ore risparmiate        -> ritorno_addestramento
  C  decisioni per episodio ed entropia delle visite MCTS                                                              -> lunghezza_entropia
  D  perdite principali (valore, ricompensa, politica; il totale e' omesso: diventa negativo per il termine di consistenza) -> perdite_principali
  E  perdite ausiliarie ed entropia della politica                                                                       -> perdite_ausiliarie
  F  norma del gradiente, learning rate, valore predetto contro valore bersaglio                                         -> gradiente_valore
Dati del pannello A (in ordine di preferenza):
  * da PRECOCI_FINO in su: la valutazione ufficiale della run, runs/<RUN>/eval_*_val.csv (seed 1000-1029, 2-3 repliche);
  * fino a PRECOCI_FINO: output/ricerca_per_iterazione_<RUN>.csv dello script 10 (100 simulazioni, 2 repliche, stessi seed), perche' la valutazione ufficiale parte da 10000. Marcatori VUOTI nel grafico.
  Lo stesso checkpoint valutato due volte sugli stessi seed puo' differire di 0.02-0.06 h per la ricerca C++ non riproducibile: dichiararlo in didascalia.
  Se uno dei due file manca il pannello usa solo l'altro; se mancano entrambi il pannello A viene saltato.
Per un'altra run cambia RUN e SCALA e, se serve, il TITOLO.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
RUN = "ft1_lungo_s0"                   # cartella in FT1/runs/
SCALA = 100                            # scala della ricompensa di MuZero: ore risparmiate = ritorno / SCALA
FINESTRA = 15                          # larghezza della media mobile nei pannelli B e C
PRECOCI_FINO = 5000                    # checkpoint <= questo valore: dall'altro protocollo (script 10), marcatori vuoti. 0 = non usarli
PANNELLI = "ABCDEF"                    # quali pannelli disegnare
INSIEME = True                         # figura d'insieme 2 x 3 (con le lettere A-F)
SEPARATI = True                        # un file per pannello (senza lettera nel titolo: pronto per essere citato dal testo)
TITOLO = ""                            # titolo della figura d'insieme ("" = nessuno)
SALVA = True
# ============================================================
import sys, re, glob, csv, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BLU, GRIGIO, ARANCIO, VERDE, VIOLA, ROSSO = "#2a78d6", "#898781", "#eb6834", "#1baf7a", "#6250d6", "#d62728"; TESTO, GRIGLIA, SFONDO = "#0b0b0b", "#e1e0d9", "#ffffff"     # v2: sfondo bianco puro (si fonde con la pagina LaTeX)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.edgecolor": "#c3c2b7", "axes.facecolor": SFONDO, "figure.facecolor": SFONDO, "savefig.facecolor": SFONDO,
                     "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8})
NOMI = {"A": "costo_checkpoint", "B": "ritorno_addestramento", "C": "lunghezza_entropia", "D": "perdite_principali", "E": "perdite_ausiliarie", "F": "gradiente_valore"}


def carica_tensorboard():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = sorted(glob.glob(str(percorsi.RUNS / RUN / "**" / "events.out.tfevents*"), recursive=True))
    if not f: raise SystemExit(f"Nessun file TensorBoard in {percorsi.RUNS / RUN}")
    ea = EventAccumulator(f[-1], size_guidance={"scalars": 0}); ea.Reload(); print(f"TensorBoard: {f[-1]}"); return ea


def lisc(y, w):
    if len(y) < 3: return y
    w = max(1, min(w, len(y))); k = np.ones(w) / w; p = np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"); return np.convolve(p, k, mode="valid")


def dati_checkpoint():
    """{iterazione: (costo MuZero per seed, costo regola per seed, repliche, protocollo)}; vedi la descrizione in cima."""
    righe = {}; reg = {}
    ufficiale = sorted(glob.glob(str(percorsi.RUNS / RUN / "eval_*_val.csv")))
    if ufficiale:
        ds = {}
        for r in csv.DictReader(open(ufficiale[0])):
            if r["who"] == "regola": reg.setdefault(int(r["seed"]), float(r["cost"]))
            else: ds.setdefault(int(re.sub(r"\D", "", r["who"])), {}).setdefault(int(r["seed"]), []).append(float(r["cost"]))
        for it, d in ds.items():
            sd = sorted(s for s in d if s in reg)
            if sd: righe[it] = (np.array([np.mean(d[s]) for s in sd]), np.array([reg[s] for s in sd]), max(len(d[s]) for s in sd), "ufficiale")
    else: print("Valutazione ufficiale (runs/<RUN>/eval_*_val.csv) non trovata")
    f10 = percorsi.OUTPUT / f"ricerca_per_iterazione_{RUN}.csv"
    if f10.exists() and PRECOCI_FINO > 0 and reg:
        ds = {}
        for r in csv.DictReader(f10.open()):
            it = int(r["it"])
            if int(r["sims"]) == 100 and it <= PRECOCI_FINO and int(r["seed"]) in reg: ds.setdefault(it, {}).setdefault(int(r["seed"]), []).append(float(r["cost"]))
        for it, d in ds.items():
            sd = sorted(s for s, v in d.items() if len(v) >= 2)                                  # solo seed con tutte e due le repliche
            if sd and it not in righe: righe[it] = (np.array([np.mean(d[s]) for s in sd]), np.array([reg[s] for s in sd]), 2, "script10")
    elif PRECOCI_FINO > 0 and not f10.exists(): print(f"Checkpoint precoci: {f10.name} non trovato (script 10)")
    return righe


class Dati:
    def __init__(self):
        self.ea = carica_tensorboard(); self.T = set(self.ea.Tags()["scalars"]); self.ck = dati_checkpoint()

    def S(self, tag):
        s = self.ea.Scalars(tag); return np.array([x.step for x in s]), np.array([x.value for x in s])

    def ha(self, tag): return tag in self.T


def pannello_A(ax, D, tit):
    righe = D.ck
    if not righe: ax.axis("off"); print("Pannello A saltato: nessun dato di valutazione dei checkpoint"); return False
    its = sorted(righe); m = np.array([righe[i][0].mean() for i in its]); r = np.array([righe[i][1].mean() for i in its]); ic = []
    for i in its:
        d = righe[i][0] - righe[i][1]; bt = np.random.default_rng(0).integers(0, len(d), (4000, len(d))); mm = d[bt].mean(1); ic.append((np.percentile(mm, 2.5), np.percentile(mm, 97.5)))
    ic = np.array(ic); xs = np.array(its); k0 = 1 if its[0] == 0 else 0; rm = float(np.mean(r))
    lo_all = np.concatenate([r[k0:] + ic[k0:, 0], m[k0:]]); hi_all = np.concatenate([r[k0:] + ic[k0:, 1], r[k0:]]); mg = 0.12 * (hi_all.max() - lo_all.min()); ylo, yhi = lo_all.min() - mg, hi_all.max() + 2.2 * mg
    ax.axhline(rm, color=GRIGIO, ls="--", lw=1.4, label=f"rule ({rm:.2f} h)")
    ax.fill_between(xs[k0:], (r + ic[:, 0])[k0:], (r + ic[:, 1])[k0:], color=BLU, alpha=0.18, label="95% CI of the paired difference"); ax.plot(xs[k0:], m[k0:], "-", color=BLU, lw=2, label="MuZero (checkpoint)")
    for i, y in zip(its, m):
        if i == 0 and k0: continue
        ax.plot([i], [y], "o", color=BLU, ms=5, markerfacecolor=("white" if righe[i][3] == "script10" else BLU), markeredgewidth=1.5)
    if k0:                                                                                       # iterazione 0: spesso fuori scala (rete non addestrata), segnata con un triangolo
        if m[0] > yhi: ax.plot([0], [yhi - 0.04 * (yhi - ylo)], "^", color=BLU, ms=7, clip_on=False); ax.text(120, yhi - 0.015 * (yhi - ylo), f"it 0: {m[0]:.2f} h ({100 * (m[0] / rm - 1):+.0f}%), off scale", fontsize=8, color=BLU, va="top")
        else: ax.plot([0], [m[0]], "o", color=BLU, ms=5)
    for i, y in zip(its, m):
        if i > 0 and (i in (1000, 3000, 10000) or i == its[-1]): ax.annotate(f"{100 * (y / rm - 1):+.1f}%", (i, y), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5, color=TESTO)
    salta = {i for i in its if i not in (0, 1000, 2000, 3000, 5000, 10000, 30000, 56880, its[-1])}   # etichette solo per alcuni checkpoint: gli altri hanno solo il tratto
    ax.set_ylim(ylo, yhi); ax.set_xscale("symlog", linthresh=1000, linscale=0.3); ax.set_xticks(its); ax.set_xticklabels(["" if i in salta else (f"{i // 1000}k" if i >= 1000 else str(i)) for i in its]); ax.minorticks_off()
    ax.set_xlabel("Learner iteration (log scale above 1k)"); ax.set_ylabel("Cost on development seeds (h)"); ax.legend(loc="lower left")
    n = len(righe[its[-1]][0]); ax.set_title(f"{tit}Checkpoint cost")
    print(f"[per la didascalia di A] {n} seed di sviluppo; marcatori vuoti = checkpoint valutati con l'altro protocollo (script 10): " + ", ".join(str(i) for i in its if righe[i][3] == "script10")); return True


def pannello_B(ax, D, tit):
    x, y = D.S("collector_iter/reward_mean"); ax.axhline(0, color=GRIGIO, ls="--", lw=1.2, label="rule (0 by construction)")
    ax.plot(x, y / SCALA, color=BLU, lw=0.6, alpha=0.25); ax.plot(x, lisc(y / SCALA, FINESTRA), color=BLU, lw=1.8, label="collection (with exploration)")
    if D.ha("evaluator_iter/eval_episode_return_mean"):
        xe, ye = D.S("evaluator_iter/eval_episode_return_mean"); ax.plot(xe, ye / SCALA, "o", color=ARANCIO, ms=4, label=f"evaluation ({int(D.S('evaluator_iter/episode_count')[1][-1])} episodes)")
    ax.set_xlabel("Learner iteration"); ax.set_ylabel(f"Hours saved vs rule (return / {SCALA:g})"); ax.set_title(f"{tit}Training return"); ax.legend(loc="lower right")


def pannello_C(ax, D, tit):
    x, y = D.S("collector_iter/avg_envstep_per_episode"); ax.plot(x, y, color=VERDE, lw=0.6, alpha=0.25); ax.plot(x, lisc(y, FINESTRA), color=VERDE, lw=1.8)
    ax.set_xlabel("Learner iteration"); ax.set_ylabel("Decisions per episode", color=VERDE)
    if D.ha("collector_iter/visit_entropy"):
        a2 = ax.twinx(); a2.grid(False); x, y = D.S("collector_iter/visit_entropy"); a2.plot(x, lisc(y, FINESTRA), color=VIOLA, lw=1.8); a2.set_ylabel("MCTS visit entropy", color=VIOLA); a2.spines["right"].set_visible(True)
    ax.set_title(f"{tit}Episode length and search entropy")


def pannello_D(ax, D, tit):
    for tag, col, lab in (("value_loss_avg", BLU, "value"), ("reward_loss_avg", ARANCIO, "reward"), ("policy_loss_avg", VERDE, "policy")):
        if D.ha(f"learner_iter/{tag}"): x, y = D.S(f"learner_iter/{tag}"); ax.plot(x, lisc(y, 7), color=col, lw=1.7, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Learner iteration"); ax.set_ylabel("Loss (log scale)"); ax.set_title(f"{tit}Main losses"); ax.legend(ncol=2)


def pannello_E(ax, D, tit):
    for tag, col, lab in (("consistency_loss_avg", VIOLA, "consistency"), ("commitment_loss_avg", ROSSO, "chance commitment"), ("afterstate_value_loss_avg", GRIGIO, "afterstate value"), ("policy_entropy_avg", ARANCIO, "policy entropy")):
        if D.ha(f"learner_iter/{tag}"): x, y = D.S(f"learner_iter/{tag}"); ax.plot(x, lisc(y, 7), color=col, lw=1.7, label=lab)
    ax.set_xlabel("Learner iteration"); ax.set_ylabel("Loss"); ax.set_title(f"{tit}Auxiliary losses and policy entropy"); ax.legend()


def pannello_F(ax, D, tit):
    if D.ha("learner_iter/total_grad_norm_before_clip_avg"): x, y = D.S("learner_iter/total_grad_norm_before_clip_avg"); ax.plot(x, lisc(y, 7), color=TESTO, lw=1.7, label="gradient norm (before clipping)")
    if D.ha("learner_iter/predicted_values_avg"):
        x, y = D.S("learner_iter/predicted_values_avg"); ax.plot(x, lisc(y, 7), color=BLU, lw=1.5, label="predicted value"); x, y = D.S("learner_iter/target_value_avg"); ax.plot(x, lisc(y, 7), color=ARANCIO, lw=1.5, ls="--", label="target value")
    if D.ha("learner_iter/cur_lr_avg"):
        a3 = ax.twinx(); a3.grid(False); x, y = D.S("learner_iter/cur_lr_avg"); a3.plot(x, y, color=GRIGIO, lw=1.2, ls=":"); a3.set_ylabel("learning rate", color=GRIGIO); a3.spines["right"].set_visible(True)
    ax.set_xlabel("Learner iteration"); ax.set_title(f"{tit}Gradient, learning rate, predicted vs target value"); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.30))


FUNZ = {"A": pannello_A, "B": pannello_B, "C": pannello_C, "D": pannello_D, "E": pannello_E, "F": pannello_F}


def salva(fig, base):
    for e in ("png", "pdf"): fig.savefig(f"{base}.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def principale():
    D = Dati(); P = [p for p in PANNELLI.upper() if p in FUNZ]; cartella = percorsi.OUTPUT / f"curve_{RUN}"
    if SALVA and SEPARATI: cartella.mkdir(exist_ok=True)
    if INSIEME and P:
        fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.2)); axs = axs.ravel()
        for ax in axs[len(P):]: ax.axis("off")
        for ax, p in zip(axs, P): FUNZ[p](ax, D, f"{p}. ")
        if TITOLO: fig.suptitle(TITOLO, x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97 if TITOLO else 1))
        if SALVA: salva(fig, percorsi.OUTPUT / f"curve_addestramento_{RUN}"); print(f"Salvata la figura d'insieme: {percorsi.OUTPUT}/curve_addestramento_{RUN}.png e .pdf")
    if SEPARATI:
        for p in P:
            fig, ax = plt.subplots(figsize=(6.4, 4.2)); FUNZ[p](ax, D, ""); fig.tight_layout()
            if SALVA: salva(fig, cartella / NOMI[p])
        if SALVA: print(f"Salvati i pannelli singoli in {cartella}/ ({', '.join(NOMI[p] for p in P)})")


if __name__ == "__main__":
    principale()
