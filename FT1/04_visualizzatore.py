"""04 - VISUALIZZATORE interattivo (HTML): replay dello STESSO episodio giocato da regola, PPO e MuZero, affiancati, con la rampa di L.

Come si usa: premi play. Sceglie l'esperimento (cartella in output/ con tracce.pkl), genera output/<esperimento>/visualizzatore.html e lo apre nel browser.
Il file HTML e' autonomo (un solo file, nessuna installazione): puoi mandarlo, aprirlo su un altro computer o usarlo durante la discussione.
Per cambiare cosa mostra: modifica lib/ft1_viewer_template.html (aspetto) e lib/ft1_viewer.py (dati). Dimmi cosa vuoi mostrare e lo aggiorno io."""
# ============================================================ IMPOSTAZIONI RAPIDE
APRI_BROWSER = True        # True = apre l'HTML nel browser predefinito appena generato
LIMITA_SEED = None         # None = tutti i seed delle tracce | numero = solo i primi N (HTML piu' leggero)
# ============================================================
import sys, pathlib, webbrowser
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib")); import percorsi
import ft1_config as C
from ft1_viewer import genera

if __name__ == "__main__":
    cart = C.scegli_esperimento("tracce.pkl")
    out, mb = genera(cart / "tracce.pkl", cart / "visualizzatore.html", LIMITA_SEED)
    print(f"\nScritto {out}  ({mb:.1f} MB)")
    if APRI_BROWSER: webbrowser.open(out.as_uri()); print("Aperto nel browser.")
