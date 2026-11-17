"""09 - LA RICERCA AIUTA? MuZero con la sola rete (nessuna ricerca) contro MuZero con 5, 25 e 100 simulazioni, sugli STESSI episodi.

Come si usa: apri questo file in VS Code e premi play (serve un ambiente Python con torch + LightZero).
Prima serve avere in output/<esperimento>/tracce.pkl le tracce di MuZero (script 02: giocatore mz). Il livello "100 simulazioni" e' quello gia' registrato in quelle tracce.

Cosa risponde:
  [SE]    Il costo medio cambia quando si toglie (o si riduce) la ricerca? -> costo per livello di simulazioni, con differenza appaiata rispetto alla regola, a PPO e a MuZero con 100 simulazioni.
          "0 simulazioni" = si gioca l'azione piu' probabile della sola rete di politica (prior mascherato con le azioni legali), senza MCTS.
  [COME]  Dove e come la ricerca cambia la scelta della rete? Sugli stati realmente visitati da MuZero (100 simulazioni) si confronta l'azione scelta dalla ricerca con quella
          che sceglierebbe la sola rete: quota di decisioni cambiate, per tipo (piazzamento / mossa di L), per fase dell'episodio, e come cambia il piazzamento (altro job, altra sezione).
Cosa NON risponde: se la ricerca serve in ADDESTRAMENTO (le visite di MCTS sono il bersaglio della politica). Quello richiede un secondo addestramento.
La ricerca C++ non e' riproducibile: due esecuzioni con le stesse simulazioni danno costi leggermente diversi (usare REPLICHE > 1 per misurarlo).
Scrive in output/<esperimento>/: ricerca_vs_rete.csv (costo per livello/seed, riprendibile), ricerca_decisioni.csv, figure/ricerca_vs_rete.png e .pdf.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
SIMULAZIONI = (0, 5, 25)   # livelli da rigiocare (0 = sola rete). Il livello 100 viene dalle tracce, a meno di RIGIOCA_100
RIGIOCA_100 = False        # True = rigioca anche 100 simulazioni (seconda replica: misura il rumore della sola ricerca)
REPLICHE = 1               # repliche per ogni livello rigiocato (per 0 simulazioni basta 1: e' deterministico)
SEED_DA, SEED_A = 9000, 9100
ANALISI_DECISIONI = True   # [COME]: confronto decisione per decisione sugli stati visitati da MuZero (veloce: nessuna ricerca)
SALVA = True               # salva CSV e figura
# ============================================================
import sys, os, time, csv, pickle, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import ft1_config as C, ft1_giocatori as G
from ft1lib.factory_ft1 import make_env

NOMI = {0: "net only (0 sims)"}
BLU, ARANCIO, GRIGIO, VERDE, VIOLA = "#2a78d6", "#eb6834", "#898781", "#1baf7a", "#6250d6"; TESTO, GRIGLIA, SFONDO = "#0b0b0b", "#e1e0d9", "#fcfcfb"


def boot(d, B=10000, seed=20261109):
    d = np.asarray(d, float); b = np.random.default_rng(seed).integers(0, len(d), (B, len(d))); m = d[b].mean(1); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agente(ck, sims):
    """Agente LightZero con 'sims' simulazioni (la rete e' sempre la stessa; per 0 si usa il prior, quindi basta costruirlo con 1)."""
    os.environ["MZ_SIMS"] = str(max(1, int(sims)))
    import torch; torch.set_num_threads(1)
    from mz_eval_ft1 import MZFTAgent
    return MZFTAgent(str(ck), "muzero")


def prior(ag, e):
    """Azione piu' probabile della sola rete di politica (logit mascherati con le azioni legali) e sua distribuzione. Nessuna ricerca."""
    import torch
    obs = torch.tensor(e._build_observation()[None], dtype=torch.float32); mask = e.action_masks().astype(bool); m = ag.policy._eval_model; m.eval()
    with torch.no_grad(): lg = m.initial_inference(obs).policy_logits[0].detach().cpu().numpy().astype(float)
    lg = np.where(mask, lg, -np.inf); p = np.exp(lg - lg.max()); p = p / p.sum(); return int(np.argmax(lg)), p


def gioca(ag, kw, s, sims):
    """Un episodio; costo in ore come nelle tracce (somma di -reward/3600)."""
    G.seed_all(s); e = make_env(dict(kw)); e.reset(seed=s); cost = 0.0; n = 0
    while True:
        if sims == 0: a = prior(ag, e)[0]
        else:
            import torch
            obs = torch.tensor(e._build_observation()[None], dtype=torch.float32); mask = e.action_masks().astype(np.int8)
            with torch.no_grad(): a = int(ag.policy.eval_mode.forward(obs, [mask], to_play=[-1], ready_env_id=np.array([0]), timestep=[0])[0]["action"])
        _, r, te, tr, _ = e.step(int(a)); cost -= r / 3600.0; n += 1
        if te or tr: return cost
        assert n < 1000


def tipo(a, na):
    return "L move" if a < na else "placement"


def decisioni(ag, kw, R, seeds, na=3):
    """Riesegue le azioni registrate di MuZero (100 simulazioni) e a ogni stato chiede alla sola rete cosa sceglierebbe."""
    rows = []
    for s in seeds:
        tr = R["mz"][s]; n = len(tr["rows"]); e = make_env(dict(kw)); e.reset(seed=s)
        for r in tr["rows"]:
            assert int(e.t) == int(r["t"]) and bool((e.action_masks() == r["mask"]).all()), f"la traccia del seed {s} non coincide con l'ambiente (config diversa?)"
            ap, p = prior(ag, e); a = int(r["a"]); v = np.asarray(r["info"].get("visits", np.zeros(len(p))), float)
            rows.append(dict(seed=s, p=r["i"] / n, a=a, ap=ap, over=int(a != ap), tipo=tipo(a, na), tipo_p=tipo(ap, na), v_prior=float(v[ap]) if len(v) > ap else np.nan, p_prior_a=float(p[a]), p_max=float(p[ap])))
            e.step(a)
    return rows


def tabella(costi, ref, nomi_liv):
    """costi: {livello: {seed: costo}}; ref: {nome: {seed: costo}}. Stampa e restituisce le righe."""
    print(f"\n{'':22}{'cost (h)':>10}  {'vs rule':>24}  {'vs PPO':>24}  {'vs MuZero 100 sims':>24}")
    out = []
    for lv in nomi_liv:
        c = costi[lv]; S = sorted(c); x = np.array([c[s] for s in S]); r = {"livello": lv, "n": len(S), "cost": float(x.mean())}; txt = f"{nomi_liv[lv]:22}{x.mean():10.3f}"
        for nm in ("rule", "ppo", "mz"):
            if nm in ref and all(s in ref[nm] for s in S):
                d = x - np.array([ref[nm][s] for s in S]); m, lo, hi = boot(d); r[nm] = (m, lo, hi); txt += f"  {m:+7.3f} [{lo:+.3f},{hi:+.3f}]"
            else: txt += f"  {'-':>24}"
        print(txt); out.append(r)
    return out


def figura(cart, righe, ref, dec, nomi_liv):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.facecolor": SFONDO, "figure.facecolor": SFONDO, "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": GRIGLIA, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": SFONDO})
    fig, ax = plt.subplots(1, 2 if dec else 1, figsize=(10.5 if dec else 5.4, 3.8)); ax = np.atleast_1d(ax)
    x = np.arange(len(righe)); m = np.array([r["rule"][0] if "rule" in r else np.nan for r in righe]); lo = np.array([r["rule"][1] if "rule" in r else np.nan for r in righe]); hi = np.array([r["rule"][2] if "rule" in r else np.nan for r in righe])
    ax[0].errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", color=BLU, lw=2, ms=6, capsize=3, label="MuZero (paired 95% CI)")
    if "ppo" in ref and "rule" in ref:
        S = sorted(set(ref["ppo"]) & set(ref["rule"])); ax[0].axhline(np.mean([ref["ppo"][s] - ref["rule"][s] for s in S]), color=ARANCIO, ls="--", lw=1.6, label="PPO")
    ax[0].axhline(0, color=GRIGIO, lw=1.2, label="rule"); ax[0].set_xticks(x); ax[0].set_xticklabels([nomi_liv[r["livello"]] for r in righe]); ax[0].set_ylabel("Cost minus rule cost (h)"); ax[0].set_title("A. Cost vs search budget", loc="left"); ax[0].legend()
    if dec:
        p = np.array([d["p"] for d in dec]); B = np.minimum(p * 4, 3.999).astype(int); o = np.array([d["over"] for d in dec]); tp = np.array([d["tipo"] for d in dec]); w = 0.38
        for k, (t, col) in enumerate((("placement", VERDE), ("L move", VIOLA))):
            v = [100 * ((o == 1) & (tp == t) & (B == q)).sum() / max(1, (B == q).sum()) for q in range(4)]; ax[1].bar(np.arange(4) + (k - 0.5) * w, v, w, color=col, label=t)
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["1st quarter", "2nd", "3rd", "4th"]); ax[1].set_ylabel("Decisions where search overrides the network (%)"); ax[1].set_title(f"B. Where search changes the choice ({100 * o.mean():.0f}% overall)", loc="left"); ax[1].legend()
    fig.tight_layout(); d = cart / "figure"; d.mkdir(exist_ok=True)
    for f in ("png", "pdf"): fig.savefig(d / f"ricerca_vs_rete.{f}", dpi=200, bbox_inches="tight")
    print(f"\nFigura salvata in {d}/ricerca_vs_rete.png e .pdf")


def principale():
    cart = C.scegli_esperimento("tracce.pkl"); d = pickle.load(open(cart / "tracce.pkl", "rb")); R, meta = d["res"], d.get("meta", {})
    if "mz" not in R or not R["mz"]: raise SystemExit("Nelle tracce non c'e' MuZero: lancia prima 02_esegui_politiche.py (giocatore mz).")
    kw = meta.get("env_kw"); ck = G.percorso(meta.get("mz_checkpoint", "modelli/muzero_ft1_it56880.pth.tar"))
    if not ck.exists(): raise SystemExit(f"Non trovo il checkpoint MuZero: {ck}")
    seeds = sorted(s for s in R["mz"] if SEED_DA <= s < SEED_A); ref = {p: {s: R[p][s]["total"] for s in R[p]} for p in ("rule", "ppo", "mz") if p in R}
    print(f"Esperimento {cart.name}: {len(seeds)} seed in comune, checkpoint {ck.name}"); na = int(R["mz"][seeds[0]].get("na", 3))
    nomi_liv = {0: "net only (0 sims)"}; nomi_liv.update({n: f"{n} sims" for n in set(SIMULAZIONI) | {100} if n}); livelli = sorted(set(SIMULAZIONI) | {100})
    f = cart / "ricerca_vs_rete.csv"; costi = {lv: {} for lv in livelli}; cont = {}
    if f.exists():
        for r in csv.DictReader(f.open()): cont.setdefault((int(r["sims"]), int(r["seed"])), []).append(float(r["cost"]))
    costi[100] = {s: R["mz"][s]["total"] for s in seeds}                                    # 100 simulazioni = tracce gia' registrate
    da_giocare = [lv for lv in livelli if lv != 100 or RIGIOCA_100]
    if RIGIOCA_100: costi[100] = {}
    ag1 = None
    for lv in da_giocare:
        todo = [s for s in seeds if len(cont.get((lv, s), [])) < (1 if lv == 0 else REPLICHE)]
        if todo:
            ag = agente(ck, lv); ag1 = ag1 or ag; t0 = time.time(); print(f"\n{nomi_liv[lv]}: {len(todo)} episodi da giocare", flush=True)
            for i, s in enumerate(todo):
                for _ in range(1 if lv == 0 else REPLICHE - len(cont.get((lv, s), []))):
                    c = gioca(ag, kw, s, lv); cont.setdefault((lv, s), []).append(c)
                    with f.open("a", newline="") as fh:
                        w = csv.writer(fh)
                        if f.stat().st_size == 0: w.writerow(["sims", "replica", "seed", "cost"])
                        w.writerow([lv, len(cont[(lv, s)]) - 1, s, f"{c:.6f}"])
                if i in (0, 4) or (i + 1) % 20 == 0: print(f"  {i + 1}/{len(todo)}  ({(time.time() - t0) / (i + 1):.1f} s per episodio)", flush=True)
        costi[lv] = {s: float(np.mean(cont[(lv, s)])) for s in seeds if (lv, s) in cont}
    righe = tabella(costi, ref, {lv: nomi_liv[lv] for lv in livelli if costi[lv]})
    for r in righe:
        if r["livello"] == 0 and "mz" in r:
            m, lo, hi = r["mz"]; print(f"\nLettura: sola rete meno MuZero a 100 simulazioni = {m:+.3f} h, IC95 [{lo:+.3f},{hi:+.3f}] -> " + ("indistinguibile: la ricerca, in valutazione, non cambia il costo medio." if lo <= 0 <= hi else ("la sola rete e' PEGGIORE: la ricerca migliora il costo." if m > 0 else "la sola rete e' MIGLIORE della ricerca.")))
    dec = []
    if ANALISI_DECISIONI:
        ag = ag1 or agente(ck, 1); print("\nAnalisi decisione per decisione sugli stati visitati da MuZero (100 simulazioni)...", flush=True); dec = decisioni(ag, kw, R, seeds, na)
        o = np.array([x["over"] for x in dec]); tp = np.array([x["tipo"] for x in dec]); p = np.array([x["p"] for x in dec]); B = np.minimum(p * 4, 3.999).astype(int)
        print(f"  decisioni: {len(dec)}; la ricerca sceglie un'azione DIVERSA da quella della sola rete nel {100 * o.mean():.1f}% dei casi")
        for t in ("placement", "L move"): m = tp == t; print(f"    {t:10s}: {m.sum():5d} decisioni, cambiate {100 * o[m].mean():.1f}%")
        print("    per quarto di episodio: " + "  ".join(f"Q{q + 1} {100 * o[B == q].mean():.1f}%" for q in range(4)))
        vp = np.array([x["v_prior"] for x in dec]); print(f"    quota media di visite MCTS sull'azione preferita dalla rete: {np.nanmean(vp):.2f}; probabilita' media della rete sull'azione scelta dalla ricerca: {np.mean([x['p_prior_a'] for x in dec]):.2f}")
        pl = [x for x in dec if x["over"] and x["tipo"] == "placement" and x["tipo_p"] == "placement"]
        if pl:
            ca = lambda a: (a - na) // 2; se = lambda a: (a - na) % 2
            solo_sez = sum(1 for x in pl if ca(x["a"]) == ca(x["ap"]) and se(x["a"]) != se(x["ap"])); solo_cl = sum(1 for x in pl if ca(x["a"]) != ca(x["ap"]) and se(x["a"]) == se(x["ap"])); entr = len(pl) - solo_sez - solo_cl
            print(f"    tra i {len(pl)} piazzamenti cambiati: stesso job ma altra sezione {solo_sez}, altro job nella stessa sezione {solo_cl}, entrambi diversi {entr}")
        if o.mean() > 0.7: print("  ATTENZIONE: quota di decisioni cambiate molto alta, controlla che il prior sia letto correttamente (il checkpoint usa la rete 'muzero' classica).")
        if SALVA:
            with (cart / "ricerca_decisioni.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(dec[0])); w.writeheader(); w.writerows(dec)
    if SALVA: figura(cart, righe, ref, dec, nomi_liv)


if __name__ == "__main__":
    principale()
