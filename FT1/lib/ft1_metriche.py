"""Metriche per episodio e tabelle riassuntive (con intervalli di confidenza appaiati). Solo numpy/pandas."""
import numpy as np
from hpcpower.env_ft import SECTIONS
PMAX = 12000.0

KEYS = ["cost", "flow_term", "energy_term", "makespan_h", "E_kWh", "spreco_kWh", "quota_usata", "L_medio", "L_min", "quota_L_sotto1", "cambi_L", "quota_tick_limitati", "flusso_medio_h", "flusso_p90_h", "flusso_max_h",
        "attesa_media_h", "slowdown_medio", "slowdown_compute", "slowdown_memory", "compute_su_veloce", "memory_su_efficiente"]


def metriche_da_traccia(tr):
    """Stesse metriche di ep_metrics ma calcolate da una traccia gia' registrata (tr = res[giocatore][seed]) senza rigiocare l'episodio."""
    dt = tr["dt"]; T = tr["ticks"]; jobs = tr["jobs"]; n = max(len(T), 1)
    E_del = sum(t["deliv"] for t in T) * dt; E_used = sum(t["used"] for t in T) * dt
    fl = np.array([(j["end"] - j["arr"]) * dt for j in jobs]) / 3600.0; wt = np.array([(j["start"] - j["arr"]) * dt for j in jobs]) / 3600.0
    ideal = np.array([j["nom"] / SECTIONS[j["sec"]][2][j["typ"]] for j in jobs]); run = np.array([(j["end"] - j["start"]) * dt for j in jobs]); slow = run / np.maximum(ideal, 1e-9)
    typ = np.array([j["typ"] for j in jobs]); sec = np.array([j["sec"] for j in jobs]); nan = np.nan
    return dict(cost=tr["total"], flow_term=tr["flow"], energy_term=tr["energy"], makespan_h=tr["makespan_h"], n_tick=len(T), E_kWh=E_del / 3.6e6, E_usata_kWh=E_used / 3.6e6, spreco_kWh=(E_del - E_used) / 3.6e6,
                quota_usata=E_used / max(E_del, 1e-9), L_medio=sum(t["L"] for t in T) / n, L_min=min([t["L"] for t in T] or [1.0]), quota_L_sotto1=sum(t["L"] < 1 - 1e-9 for t in T) / n,
                cambi_L=sum(abs(t["L"] - t.get("Lprev", t["L"])) > 1e-9 for t in T) / n, quota_tick_limitati=sum(t["D"] > t["deliv"] + 1e-9 for t in T) / n,
                flusso_medio_h=fl.mean(), flusso_p90_h=np.percentile(fl, 90), flusso_max_h=fl.max(), attesa_media_h=wt.mean(), slowdown_medio=slow.mean(),
                slowdown_compute=slow[typ == 0].mean() if (typ == 0).any() else nan, slowdown_memory=slow[typ == 1].mean() if (typ == 1).any() else nan,
                compute_su_veloce=np.mean(sec[typ == 0] == 0) if (typ == 0).any() else nan, memory_su_efficiente=np.mean(sec[typ == 1] == 1) if (typ == 1).any() else nan)


def ep_metrics(e, act):
    """Gioca un episodio gia' resettato con act(e) -> azione e restituisce le metriche (usato per le regole, senza registrare tracce)."""
    from ft1_traccia import play
    tr = play(e, lambda env: (act(env), {}), None); return metriche_da_traccia(tr)


def boot_ci(d, B=10000, seed=20261108):
    rng = np.random.default_rng(seed); bt = rng.integers(0, len(d), (B, len(d))); m = d[bt].mean(1); return np.percentile(m, [2.5, 97.5])


def tabella_costi(res, players, ref="rule"):
    """Costo medio per politica; differenza % rispetto a ref con IC95% appaiato (stessi seed). Righe: lista di dict."""
    seeds = sorted(set.intersection(*[set(res[p]) for p in players])); out = []
    base = np.array([res[ref][s]["total"] for s in seeds]) if ref in res else None
    for p in players:
        c = np.array([res[p][s]["total"] for s in seeds]); r = dict(player=p, n=len(seeds), cost=c.mean(), sd=c.std(ddof=1) if len(c) > 1 else np.nan)
        if base is not None and p != ref:
            d = c - base; lo, hi = boot_ci(d); r.update(rel=100 * d.mean() / base.mean(), rel_lo=100 * lo / base.mean(), rel_hi=100 * hi / base.mean(), better=int((d < 0).sum()))
        out.append(r)
    return out


def tabella_metriche(res, players, keys=None):
    """Media di ogni metrica per politica (da tracce). Restituisce DataFrame (righe = metriche, colonne = politiche)."""
    import pandas as pd
    keys = keys or KEYS; seeds = sorted(set.intersection(*[set(res[p]) for p in players])); cols = {}
    for p in players:
        M = [metriche_da_traccia(res[p][s]) for s in seeds]; cols[p] = {k: float(np.nanmean([m[k] for m in M])) for k in keys}
    return pd.DataFrame(cols)
