"""Piccole utilita' per i menu da terminale (input numerico)."""


def _leggi(prompt):
    try: return input(prompt).strip()
    except EOFError: raise SystemExit("\n(input terminato)")


def chiedi(prompt, default=None, tipo=str):
    """Chiede un valore; Invio = default. tipo: str | int | float."""
    while True:
        r = _leggi(f"{prompt}" + (f" [Invio = {default}]" if default is not None else "") + ": ")
        if r == "" and default is not None: return default
        try: return tipo(r)
        except ValueError: print("  valore non valido, riprova")


def si_no(prompt, default=True):
    r = _leggi(f"{prompt} [{'S/n' if default else 's/N'}]: ").lower()
    return default if r == "" else r.startswith(("s", "y"))


def scegli(titolo, voci, multi=False, default=None, tutte=False):
    """Mostra un elenco numerato e restituisce l'indice (o la lista di indici se multi).
    voci: lista di stringhe. default: indice/indici scelti con Invio. multi: accetta '1,3,5' o '2-4'. tutte: in multi accetta 't' = tutte."""
    print(f"\n{titolo}")
    for i, v in enumerate(voci, 1): print(f"  {i:2d}) {v}")
    dflt = None if default is None else (",".join(str(d + 1) for d in default) if multi else str(default + 1))
    while True:
        r = _leggi("Scelta" + (" (es. 1,3 oppure 2-4" + (", t = tutte" if tutte else "") + ")" if multi else "") + (f" [Invio = {dflt}]" if dflt else "") + ": ").lower()
        if r == "" and default is not None: return default
        try:
            if not multi:
                k = int(r) - 1
                if 0 <= k < len(voci): return k
            else:
                if tutte and r in ("t", "tutte", "a"): return list(range(len(voci)))
                out = []
                for p in r.replace(" ", "").split(","):
                    if "-" in p: a, b = p.split("-"); out += list(range(int(a) - 1, int(b)))
                    else: out.append(int(p) - 1)
                if out and all(0 <= k < len(voci) for k in out): return sorted(set(out))
        except ValueError: pass
        print("  scelta non valida, riprova")
