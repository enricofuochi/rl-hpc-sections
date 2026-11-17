"""CONFIGURAZIONE DI RIFERIMENTO DI FT1  --  NON MODIFICARE QUESTO FILE.

Sono i parametri con cui sono stati addestrati PPO e MuZero e con cui sono stati ottenuti i risultati della tesi (identici a ft1lib/config_ft1.py).
Per fare prove con parametri diversi usa  config/config_utente.py .

Ogni riga e' del tipo   NOME = valore   # spiegazione.   Etichette:
  [OK agenti]  cambiarlo non rompe PPO/MuZero (ma i risultati sono fuori dalla distribuzione di addestramento, quindi esplorativi)
  [ROMPE]      PPO/MuZero non funzionano piu' (cambia la forma di osservazione o azioni): usalo solo con le regole
"""

# ------------------------------------------------------------------ ambiente (FT1Env = FT0Env + livello di potenza L come stato)
N_JOB = 24                  # [OK agenti] job in coda al tick 0 (tutti presenti dall'inizio, nessun arrivo successivo)
DT = 600.0                  # [OK agenti] durata di un tick in secondi (10 minuti)
W_ENERGIA = 5.7             # [OK agenti, ma cambia la funzione di costo] peso dell'energia nel costo: costo = flusso medio (h) + w * ore-equivalenti a piena potenza / N
FONDO = (0.70, 0.80)        # [OK agenti] frazione di nodi gia' occupata da job di background all'inizio, estratta uniforme in questo intervallo
DUR_MEDIANA = 1800.0        # [OK agenti] mediana (s) della durata nominale dei job (lognormale), prima di scalare con la taglia
SIGMA = 1.0                 # [OK agenti] sigma della lognormale delle durate (variabilita' delle durate)
DELTA = 0.1                 # [OK agenti, fuori distribuzione] di quanto puo' cambiare L in un tick: le tre mosse di avanzamento sono L - DELTA, L, L + DELTA
LMIN = 0.5                  # [OK agenti, fuori distribuzione] livello minimo di L (il massimo e' 1.0 = potenza installata 12 kW; L parte da 1.0)
MAX_TICK = 400              # [OK agenti] sicurezza: tick massimi per episodio
QK = 24                     # [ROMPE] quanti job della coda compaiono nell'osservazione (146 numeri con 24)

# ------------------------------------------------------------------ che cosa far girare
SEED_DA = 9000              # primo seed (incluso). 9000-9099 sono i seed "freschi" usati per i risultati finali di FT1
SEED_A = 9100               # ultimo seed (escluso)
GIOCATORI = ("rule", "ppo", "mz")   # tra: rule (regola con tracking di L), l1 (stessa regola con L = 1 sempre), rand (casuale), plan1/plan2 (pianificatori PRIVILEGIATI, 1 o 2 passi), ppo, mz (MuZero)
REGOLA = "ft1_sjf_trk05"    # regola scritta di riferimento: ordine (fifo|sjf|lpt) + livello (L1 = sempre 1.0 | trk00|trk05|trk10|trk20 = tracking di L* = D/Pmax + margine). E' anche la regola del reward di scarto
CASI_N_SEED = 60            # seed per ciascuno scenario nella modalita' "scenari" (9000 .. 9000+N-1)

# ------------------------------------------------------------------ modelli salvati (percorsi relativi alla cartella FT1)
PPO_CHECKPOINT = "modelli/ppo_ft1_best_r0"                   # senza .zip; accanto c'e' _obsrms.pkl (PPO FT1: nessuna normalizzazione delle osservazioni)
MUZERO_CHECKPOINT = "modelli/muzero_ft1_it56880.pth.tar"     # MuZero classico, iterazione 56880 (ultimo checkpoint della run ft1_lungo_s0)
MZ_SIMULAZIONI = 100        # simulazioni MCTS per decisione (come in addestramento)

# ------------------------------------------------------------------ nome dell'esperimento (cartella in output/)
NOME_ESPERIMENTO = "riferimento"

# ------------------------------------------------------------------ scenari. Caso 1 = distribuzione di addestramento, gli altri sono fuori distribuzione
CASI = {
    1: dict(pf=(0.70, 0.80), dur_med=1800.0),
    2: dict(pf=(0.40, 0.50), dur_med=1800.0),
    3: dict(pf=(0.85, 0.95), dur_med=1800.0),
    4: dict(pf=(0.70, 0.80), dur_med=900.0),
    5: dict(pf=(0.70, 0.80), dur_med=3600.0),
    6: dict(pf=(0.85, 0.95), dur_med=3600.0),
}
