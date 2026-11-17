"""04 - TABELLE dei risultati: costi (con intervalli di confidenza appaiati) e metriche generali (makespan, energia, flusso, attese...), per politica.

Come si usa: premi play. Sceglie l'esperimento e stampa le tabelle. Con SALVA_TABELLE = True scrive anche CSV e LaTeX in output/<esperimento>/tabelle/.
Le tabelle si calcolano dalle tracce gia' registrate (nessun episodio viene rigiocato). Per gli scenari (casi.pkl) fa una tabella per caso."""
# ============================================================ IMPOSTAZIONI RAPIDE
SALVA_TABELLE = False      # True = salva CSV + LaTeX in output/<esperimento>/tabelle/
RIFERIMENTO = "rule"       # politica rispetto a cui si calcolano le differenze percentuali
DECIMALI = 3
# metriche mostrate nella tabella generale (nome interno -> intestazione). Togli o riordina le righe a piacere.
METRICHE = {"cost": "Cost (h)", "flow_term": "Flow term (h)", "energy_term": "Energy term (h)", "makespan_h": "Makespan (h)", "E_kWh": "Energy delivered (kWh)", "spreco_kWh": "Delivered, unused (kWh)",
            "quota_usata": "Delivered power used (frac.)", "L_medio": "Mean level L", "quota_L07": "Ticks at L=0.7 (frac.)", "flusso_medio_h": "Mean flow time (h)", "flusso_p90_h": "P90 flow time (h)",
            "attesa_media_h": "Mean wait (h)", "slowdown_medio": "Mean slowdown", "compute_su_veloce": "Compute on fast (frac.)", "memory_su_efficiente": "Memory on efficient (frac.)"}
# ============================================================
import sys, pickle, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np, pandas as pd
import ft0_config as C
from ft0_metriche import tabella_costi, tabella_metriche
NOMI = dict(rule="Rule (no waiting)", rulew="Rule (waiting)", rand="Random", ppo="PPO", mz="MuZero"); ORDINE = ["rand", "rule", "rulew", "ppo", "mz"]


def salva(df, cart, nome, **kw):
    if not SALVA_TABELLE: return
    d = cart / "tabelle"; d.mkdir(exist_ok=True); df.to_csv(d / f"{nome}.csv"); (d / f"{nome}.tex").write_text(df.to_latex(float_format=lambda x: f"{x:.{DECIMALI}f}", **kw)); print(f"  salvata tabelle/{nome}.csv e .tex")


def tabelle_tracce(cart):
    R = pickle.load(open(cart / "tracce.pkl", "rb"))["res"]; pol = [p for p in ORDINE if p in R and R[p]]; ref = RIFERIMENTO if RIFERIMENTO in pol else pol[0]
    rows = tabella_costi(R, pol, ref); df = pd.DataFrame(rows).set_index("player"); out = pd.DataFrame(index=[NOMI[p] for p in df.index])
    out["Seeds"] = df["n"].values; out["Cost mean (h)"] = df["cost"].values; out["Cost sd"] = df["sd"].values
    if "rel" in df: out[f"vs {NOMI[ref]} (%)"] = df["rel"].values; out["95% CI low"] = df["rel_lo"].values; out["95% CI high"] = df["rel_hi"].values; out["Seeds better"] = df["better"].values
    print(f"\n=== Costo per politica (media sui seed in comune; differenze appaiate, IC95% bootstrap) ===\n{out.round(DECIMALI).to_string()}"); salva(out, cart, "costi")
    m = tabella_metriche(R, pol, list(METRICHE)); m.index = [METRICHE[k] for k in m.index]; m.columns = [NOMI[p] for p in m.columns]
    print(f"\n=== Metriche generali (media per episodio) ===\n{m.round(DECIMALI).to_string()}"); salva(m, cart, "metriche")
    if "mz" in R and "ppo" in R:
        seeds = sorted(set(R["mz"]) & set(R["ppo"])); d = np.array([R["mz"][s]["total"] - R["ppo"][s]["total"] for s in seeds]); from ft0_metriche import boot_ci; lo, hi = boot_ci(d)
        print(f"\nMuZero - PPO (appaiato, {len(seeds)} seed): {d.mean():+.4f} h  IC95% [{lo:+.4f}, {hi:+.4f}]   (<0: MuZero meglio)")


def tabelle_casi(cart):
    RES = {}
    for f in sorted(cart.glob("casi*.pkl")):
        for c, v in pickle.load(open(f, "rb"))["res"].items(): RES.setdefault(c, {}).update(v)
    CS = sorted(RES); pol = [p for p in ORDINE if all(p in RES[c] and RES[c][p] for c in CS)]
    for key, lab in (("total", "Cost (h)"), ("makespan_h", "Makespan (h)"), ("E_kWh", "Energy delivered (kWh)")):
        df = pd.DataFrame({NOMI[p]: [np.mean([RES[c][p][s][key] for s in RES[c][p]]) for c in CS] for p in pol}, index=[f"case {c}" for c in CS])
        print(f"\n=== Scenari: {lab} ===\n{df.round(DECIMALI).to_string()}"); salva(df, cart, f"scenari_{key}")


if __name__ == "__main__":
    cart = C.scegli_esperimento("*.pkl"); print(f"SALVA_TABELLE = {SALVA_TABELLE}")
    if (cart / "tracce.pkl").exists(): tabelle_tracce(cart)
    if list(cart.glob("casi*.pkl")): tabelle_casi(cart)
    if (cart / "regole.csv").exists(): print(f"\n=== Regole scritte (da regole.csv) ===\n{pd.read_csv(cart / 'regole.csv').round(DECIMALI).to_string(index=False)}")
