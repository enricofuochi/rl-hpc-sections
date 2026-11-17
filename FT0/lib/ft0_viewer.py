"""Genera il visualizzatore interattivo (un solo file HTML, nessuna dipendenza) a partire da tracce.pkl.
Mostra, per ogni seed, regola/PPO/MuZero affiancati nello stesso istante (nodi per sezione, coda, livello L, potenza, costo cumulato) e, decisione per decisione,
cosa preferivano PPO (probabilita') e MuZero (visite della ricerca). Il template HTML/JS e' lib/ft0_viewer_template.html (e' quello che si modifica per l'aspetto)."""
import pickle, json
import numpy as np
from pathlib import Path
from hpcpower.env_ft import SECTIONS

TEMPLATE = Path(__file__).with_name("ft0_viewer_template.html")


def _pack(tr):
    out = {}
    for sec in (0, 1):
        cap = SECTIONS[sec][0]; free = np.zeros(cap)
        for j in sorted([j for j in tr["jobs_bg"] + tr["jobs"] if j["sec"] == sec], key=lambda j: (j["start"], j["id"])):
            idx = [k for k in range(cap) if free[k] <= j["start"]][:j["n"]]
            for k in idx: free[k] = j["end"]
            out[j["id"]] = idx
    return out


def _mask_int(m): return int(sum(1 << i for i, v in enumerate(m) if v))
def _pct(x): return [int(round(float(v) * 100)) for v in x]


def _pack_trace(p, tr):
    nd = _pack(tr); rows = []
    for r in tr["rows"]:
        ppo = r["info"]["probs"] if p == "ppo" else (r["q"]["ppo"][1].get("probs") if "ppo" in r["q"] else None)
        mz = r["info"]["visits"] if p == "mz" else (r["q"]["mz"][1].get("visits") if "mz" in r["q"] else None)
        rows.append([r["t"], r["a"], _mask_int(r["mask"]), _pct(ppo) if ppo is not None else None, _pct(mz) if mz is not None else None, round(float(r["ctg_rule"]), 3)])
    return dict(tot=round(tr["total"], 4), flow=round(tr["flow"], 4), en=round(tr["energy"], 4), dt=tr["dt"], mk=round(tr["makespan_h"], 3),
                jobs=[[j["typ"], j["sec"], j["start"], j["end"], j["arr"], j["n"], nd[j["id"]], j["id"], round(j["nom"])] for j in tr["jobs"]], bg=[[j["typ"], j["sec"], j["end"], j["n"], nd[j["id"]]] for j in tr["jobs_bg"]],
                ticks=[[t["t"], t["L"], round(t["D"]), round(t["deliv"]), round(t["used"]), t["nq"], t["nsys"], round(t["flow_inc"], 5), round(t["en_inc"], 5)] for t in tr["ticks"]], dec=rows)


def genera(pkl, out_html, seed_max=None):
    """pkl: percorso di tracce.pkl. Restituisce (percorso html, dimensione MB). Il viewer usa le politiche tra rule, ppo, mz presenti."""
    d = pickle.load(open(pkl, "rb")); R = d["res"]; META = d.get("meta", {}); PL = [p for p in ("rule", "ppo", "mz") if p in R and R[p]]
    if not PL: raise SystemExit("tracce.pkl non contiene rule/ppo/mz")
    seeds = sorted(set.intersection(*[set(R[p]) for p in PL]))[:seed_max]; sel = {}
    if "mz" in PL and "ppo" in PL:
        diff = {s: R["mz"][s]["total"] - R["ppo"][s]["total"] for s in seeds}; med = float(np.median(list(diff.values())))
        sel = {"typical": min(diff, key=lambda s: abs(diff[s] - med)), "mz_best": min(diff, key=diff.get), "mz_worst": max(diff, key=diff.get)}
    data = dict(seeds=seeds, sel=sel, players=PL, meta={k: v for k, v in META.items() if k not in ("mz_keys", "env_kw")}, tr={str(s): {p: _pack_trace(p, R[p][s]) for p in PL} for s in seeds})
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    Path(out_html).write_text(html); return Path(out_html), len(html) / 1e6
