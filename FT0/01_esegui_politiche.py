"""01 - ESEGUI LE POLITICHE (regole scritte, PPO, MuZero) su episodi FT0 con i modelli salvati.

Come si usa: apri questo file in VS Code e premi il tasto play (in alto a destra). Rispondi alle domande nel terminale (numeri, Invio = valore proposto).
Cosa produce (in output/<NOME_ESPERIMENTO>/):
  [1] tracce.pkl   tracce complete decisione per decisione degli stessi episodi giocati da piu' politiche -> servono a tabelle (04), visualizzatore (03)
  [2] casi.pkl     serie temporali di sei scenari -> tabella per caso (04_tabelle.py)
  [3] regole.csv   confronto veloce di tutte le regole scritte + casuale (nessun modello, pochi secondi)
Se il file esiste gia' ti chiede se riprendere (aggiunge solo i seed/giocatori mancanti) o ricominciare.
MuZero richiede il venv con LightZero (quello del Mac). Regole e PPO girano ovunque. La ricerca MuZero non e' riproducibile: due run danno risultati leggermente diversi.
"""
# ============================================================ IMPOSTAZIONI RAPIDE
CONFIG = None      # None = chiede all'avvio | "fissa" (riferimento) | "utente" (la tua copia)
AZIONE = None      # None = chiede | 1 = tracce | 2 = scenari | 3 = regole
# ============================================================
import sys, os, time, pickle, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import numpy as np
import ft0_config as C, ft0_ui as UI, ft0_giocatori as G
from hpcpower.factory import make_env
from hpcpower.policies_ft import FT_POLICIES
from ft0_traccia import play
from ft0_serie import ep_series

TUTTI = ["rule", "rulew", "rand", "ppo", "mz"]
REGOLA_SEED7000 = 2.7472222222222196      # costo della regola ft_sjf_th80 sul seed 7000 nell'ambiente di riferimento: controllo che l'ambiente sia quello giusto


def carica_pkl(out, kw, ripresa):
    """Se out esiste chiede se riprendere. Restituisce (res, meta) da cui ripartire."""
    if out.exists():
        d = pickle.load(open(out, "rb"))
        if d.get("meta", {}).get("env_kw") not in (None, kw): print(f"\n!! {out.name} esiste ma e' stato prodotto con parametri d'ambiente DIVERSI: riprendere mescolerebbe dati non confrontabili.")
        elif UI.scegli(f"\n{out.name} esiste gia'. Cosa faccio?", ["riprendo (aggiungo solo cio' che manca)", "ricomincio da zero (sovrascrivo)"], default=0) == 0: return d["res"], d.get("meta", {})
        elif not UI.si_no("Sovrascrivo davvero?", False): raise SystemExit("Annullato.")
    return None, {}


def scegli_giocatori(cfg, titolo="Quali giocatori faccio giocare?", disponibili=TUTTI):
    voci = [f"{k:5s} {G.NOMI[k]}" for k in disponibili]; dflt = [i for i, k in enumerate(disponibili) if k in cfg["GIOCATORI"]]
    return [disponibili[i] for i in UI.scegli(titolo, voci, multi=True, default=dflt, tutte=True)]


# ------------------------------------------------------------------ [1] TRACCE
def tracce(cfg, cart):
    players = scegli_giocatori(cfg)
    a0 = UI.chiedi("Primo seed", cfg["SEED_DA"], int); a1 = UI.chiedi("Ultimo seed (escluso)", cfg["SEED_A"], int)
    kw = C.env_kw(cfg); out = cart / "tracce.pkl"
    old, meta = carica_pkl(out, kw, True); res = old or {}
    for p in players: res.setdefault(p, {})
    meta.update(env_kw=kw, regola=cfg["REGOLA"], resid_scale=float(os.environ.get("MZ_RESID_SCALE", 100.0)), mz_sims=cfg["MZ_SIMULAZIONI"], mz_checkpoint=cfg["MUZERO_CHECKPOINT"], ppo_checkpoint=cfg["PPO_CHECKPOINT"])
    W = G.costruisci(cfg, players); C.salva_config_usata(cfg, cart); controllo = C.e_riferimento(cfg) and cfg["REGOLA"] == "ft_sjf_th80"
    da_fare = [(s, p) for s in range(a0, a1) for p in players if s not in res[p]]; print(f"\nEpisodi da giocare: {len(da_fare)}", flush=True)
    t0 = time.time(); fatti = 0
    for s in range(a0, a1):
        for p in players:
            if s in res[p]: continue
            G.seed_all(s); e = make_env(dict(kw, allow_wait=G.ATTESA[p])); e.reset(seed=s)
            if hasattr(W[p], "reset"): W[p].reset(s)
            qs = {q: W[q] for q in ("ppo", "mz") if q in W and q != p} if p in ("rule", "ppo", "mz") else {}     # interrogazioni incrociate (accordo tra politiche, valore sugli stati dell'altra)
            res[p][s] = play(e, W[p], qs, rule_name=cfg["REGOLA"]); fatti += 1
            if controllo and p == "rule" and s == 7000: assert abs(res[p][s]["total"] - REGOLA_SEED7000) < 1e-9, "Controllo ambiente fallito: la regola sul seed 7000 non fa 2.7472"
        eta = (time.time() - t0) / max(fatti, 1) * (len(da_fare) - fatti) / 60
        print(f"seed {s}", {p: round(res[p][s]["total"], 3) for p in players if s in res[p]}, f"| {fatti}/{len(da_fare)} episodi, mancano ~{eta:.0f} min", flush=True)
        if (s - a0) % 5 == 4 or s == a1 - 1: pickle.dump(dict(res=res, meta=meta), open(out, "wb"))
    pickle.dump(dict(res=res, meta=meta), open(out, "wb")); print(f"\nScritto {out}")
    from ft0_metriche import tabella_costi
    ps = [p for p in TUTTI if p in res]; ref = "rule" if "rule" in res else ps[0]
    print(f"\nCosto medio (h) sui seed in comune (differenze % e IC95% appaiato rispetto a: {G.NOMI[ref]}):")
    for r in tabella_costi(res, ps, ref): print(f"  {r['player']:6s} {r['cost']:.4f}" + (f"   {r['rel']:+.1f}% [{r['rel_lo']:+.1f}, {r['rel_hi']:+.1f}] vs {ref}" if "rel" in r else "   (riferimento)"))


# ------------------------------------------------------------------ [2] SCENARI
def scenari(cfg, cart):
    players = scegli_giocatori(cfg, "Quali giocatori nei sei scenari?", ["rule", "rulew", "rand", "ppo", "mz"]); casi_cfg = cfg["CASI"]
    ck = list(casi_cfg); casi = [ck[i] for i in UI.scegli("Quali scenari?", [f"caso {c}: fondo {casi_cfg[c]['pf']}, mediana durata {casi_cfg[c]['dur_med']:.0f} s" for c in ck], multi=True, default=list(range(len(ck))), tutte=True)]
    n = UI.chiedi("Quanti seed per scenario", cfg["CASI_N_SEED"], int); a0 = cfg["SEED_DA"]
    out = cart / "casi.pkl"; kw = C.env_kw(cfg); old, meta = carica_pkl(out, kw, True)
    res = old or {}; W = G.costruisci(cfg, players); meta.update(env_kw=kw, mz_sims=cfg["MZ_SIMULAZIONI"]); C.salva_config_usata(cfg, cart)
    controllo = C.e_riferimento(cfg) and cfg["REGOLA"] == "ft_sjf_th80"; t0 = time.time()
    for c in casi:
        res.setdefault(c, {})
        for p in players: res[c].setdefault(p, {})
        for s in range(a0, a0 + n):
            for p in players:
                if s in res[c][p]: continue
                G.seed_all(s); e = make_env(dict(kw, pf=tuple(casi_cfg[c]["pf"]), dur_med=casi_cfg[c]["dur_med"], allow_wait=G.ATTESA[p])); e.reset(seed=s)
                if hasattr(W[p], "reset"): W[p].reset(s)
                res[c][p][s] = ep_series(e, lambda env, who=W[p]: who(env)[0])
                if controllo and c == 1 and p == "rule" and s == 7000: assert abs(res[c][p][s]["total"] - REGOLA_SEED7000) < 1e-9, "Controllo ambiente fallito"
            if (s - a0) % 10 == 9: print(f"caso {c} seed {s}", {p: round(res[c][p][s]["total"], 3) for p in players}, f"{(time.time() - t0) / 60:.1f} min", flush=True)
        cases = {k: dict(casi_cfg[k]) for k in res}
        pickle.dump(dict(res=res, meta=meta, cases=cases), open(out, "wb"))
    print(f"\nScritto {out}")


# ------------------------------------------------------------------ [3] REGOLE
def regole(cfg, cart):
    nomi = sorted(FT_POLICIES) + ["random"]; dflt = [i for i, k in enumerate(nomi) if not k.startswith("ftW_") or k.endswith("th80")]
    sel = [nomi[i] for i in UI.scegli("Quali regole? (ft_ = senza attesa, ftW_ = i compute aspettano la sezione veloce)", nomi, multi=True, default=dflt, tutte=True)]
    a0 = UI.chiedi("Primo seed", cfg["SEED_DA"], int); a1 = UI.chiedi("Ultimo seed (escluso)", cfg["SEED_A"], int); kw = C.env_kw(cfg); rows = []
    import pandas as pd
    for nome in sel:
        for s in range(a0, a1):
            e = make_env(dict(kw, allow_wait=nome.startswith("ftW_"))); e.reset(seed=s)
            if nome == "random": rng = np.random.default_rng(s + 7); act = lambda env: int(rng.choice(np.flatnonzero(env.action_masks())))
            else: act = FT_POLICIES[nome]
            r = ep_series(e, act); rows.append(dict(regola=nome, seed=s, cost=r["total"], flow=r["flow"], energy_term=r["energy_term"], makespan_h=r["makespan_h"], E_kWh=r["E_kWh"]))
        print(f"  {nome:14s} fatto", flush=True)
    df = pd.DataFrame(rows); df.to_csv(cart / "regole_perseed.csv", index=False); riass = df.groupby("regola")[["cost", "flow", "energy_term", "makespan_h", "E_kWh"]].mean().sort_values("cost")
    if cfg["REGOLA"] in riass.index: riass["vs_rif_%"] = 100 * (riass["cost"] / riass.loc[cfg["REGOLA"], "cost"] - 1)
    riass.to_csv(cart / "regole.csv"); print("\n" + riass.round(3).to_string()); print(f"\nScritto {cart / 'regole.csv'} (e regole_perseed.csv)")


if __name__ == "__main__":
    cfg = C.scegli_config(CONFIG); cart = C.cartella_esperimento(cfg)
    az = AZIONE or UI.scegli("Che cosa faccio girare?", ["Tracce complete (regole/PPO/MuZero, stessi episodi)  -> figure, tabelle, visualizzatore", "Scenari (sei casi)  -> tabella per caso", "Tutte le regole scritte + casuale (veloce, niente modelli)"], default=0) + 1
    {1: tracce, 2: scenari, 3: regole}[az](cfg, cart)
    print("\nFinito. Prossimi passi: 04_tabelle.py (tabelle), 03_visualizzatore.py (replay interattivo).")
