"""Caricamento delle configurazioni (config/config_fissa.py e config/config_utente.py) e costruzione dell'ambiente.
Una configurazione e' un dict con i nomi in MAIUSCOLO dei file di config."""
import importlib.util, copy, json
import percorsi
from ft1_ui import scegli

FILE = {"fissa": percorsi.CONFIG / "config_fissa.py", "utente": percorsi.CONFIG / "config_utente.py"}


def carica(nome):
    spec = importlib.util.spec_from_file_location(f"_cfg_{nome}", FILE[nome]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cfg = {k: copy.deepcopy(v) for k, v in vars(m).items() if k.isupper()}; cfg["_NOME"] = nome; return cfg


def differenze(cfg):
    """Parametri di cfg diversi dal riferimento (config_fissa)."""
    ref = carica("fissa"); return {k: (ref.get(k), v) for k, v in cfg.items() if not k.startswith("_") and ref.get(k) != v}


def scegli_config(forza=None):
    """forza: None = chiede in terminale | 'fissa' | 'utente'. Mostra le differenze rispetto al riferimento."""
    if forza is None:
        k = scegli("Quale configurazione uso?", ["fissa  (riferimento, non modificabile)", "utente (la tua copia modificabile)"], default=1); forza = ("fissa", "utente")[k]
    cfg = carica(forza); d = differenze(cfg)
    print(f"\nConfigurazione: {forza}  |  esperimento: output/{cfg['NOME_ESPERIMENTO']}/")
    if d:
        print("  Parametri diversi dal riferimento:")
        for k, (a, b) in d.items(): print(f"    {k}: {a}  ->  {b}")
    else: print("  (identica al riferimento)")
    return cfg


def env_kw(cfg, **override):
    """Argomenti per ft1lib.factory_ft1.make_env (nella configurazione fissa coincidono con ft1lib.config_ft1.KW)."""
    kw = dict(env="ft1", n_jobs=cfg["N_JOB"], dt=cfg["DT"], w=cfg["W_ENERGIA"], pf=tuple(cfg["FONDO"]), dur_med=cfg["DUR_MEDIANA"], sigma=cfg["SIGMA"],
              max_ticks=cfg["MAX_TICK"], delta=cfg["DELTA"], lmin=cfg["LMIN"], xobs=True, qk=cfg["QK"])
    kw.update(override); return kw


def e_riferimento(cfg):
    """True se l'ambiente di cfg coincide con quello di riferimento (allora valgono i controlli sui numeri noti)."""
    return env_kw(cfg) == env_kw(carica("fissa"))


def cartella_esperimento(cfg):
    p = percorsi.OUTPUT / cfg["NOME_ESPERIMENTO"]; p.mkdir(parents=True, exist_ok=True); return p


def salva_config_usata(cfg, cartella, extra=None):
    d = {k: v for k, v in cfg.items()}; d.update(extra or {}); (cartella / "config_usata.json").write_text(json.dumps(d, indent=1, default=str))


def elenco_esperimenti(file):
    """Cartelle in output/ che contengono 'file' (o un pattern, es. 'casi*.pkl')."""
    return sorted(p for p in percorsi.OUTPUT.iterdir() if p.is_dir() and list(p.glob(file)))


def scegli_esperimento(file, titolo="Su quale esperimento lavoro?"):
    es = elenco_esperimenti(file)
    if not es: raise SystemExit(f"No experiment in {percorsi.OUTPUT} contains {file}. Run 02_esegui_politiche.py first (option 1), or for a single episode: python play_episode.py --seed 9000 --html episode.html")
    if len(es) == 1: print(f"\nEsperimento: {es[0].name}"); return es[0]
    return es[scegli(titolo, [f"{p.name}" for p in es], default=0)]
