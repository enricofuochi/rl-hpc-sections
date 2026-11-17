"""CONFIGURAZIONE DI RIFERIMENTO DI FT0  --  NON MODIFICARE QUESTO FILE.

Sono i parametri con cui sono stati addestrati PPO e MuZero e con cui sono stati ottenuti i risultati della tesi.
Per fare prove con parametri diversi usa  config/config_utente.py .

Ogni riga e' del tipo   NOME = valore   # spiegazione.   Etichette:
  [OK agenti]  cambiarlo non rompe PPO/MuZero (ma i risultati sono fuori dalla distribuzione di addestramento, quindi esplorativi)
  [ROMPE]      PPO/MuZero non funzionano piu' (cambia la forma di osservazione o azioni): usalo solo con le regole
"""

# ------------------------------------------------------------------ ambiente (FT0Env)
N_JOB = 24                  # [OK agenti] job in coda al tick 0 (tutti presenti dall'inizio, nessun arrivo successivo)
DT = 600.0                  # [OK agenti] durata di un tick in secondi (10 minuti)
W_ENERGIA = 5.7             # [OK agenti, ma cambia la funzione di costo] peso dell'energia nel costo: costo = flusso medio (h) + w * ore-equivalenti a piena potenza / N
FONDO = (0.70, 0.80)        # [OK agenti] frazione di nodi gia' occupata da job di background all'inizio, estratta uniforme in questo intervallo
DUR_MEDIANA = 1800.0        # [OK agenti] mediana (s) della durata nominale dei job (lognormale), prima di scalare con la taglia
SIGMA = 1.0                 # [OK agenti] sigma della lognormale delle durate (variabilita' delle durate)
LIVELLI = (1.0, 0.7)        # [ROMPE se cambi il NUMERO di livelli] frazioni della potenza massima (P_max = 12 kW) tra cui scegliere
MAX_TICK = 400              # [OK agenti] sicurezza: tick massimi per episodio
QK = 24                     # [ROMPE] quanti job della coda compaiono nell'osservazione (146 numeri con 24)

# ------------------------------------------------------------------ che cosa far girare
SEED_DA = 7000              # primo seed (incluso). 7000-7099 sono i seed "freschi" usati per i risultati finali
SEED_A = 7100               # ultimo seed (escluso)
GIOCATORI = ("rule", "ppo", "mz")   # tra: rule (regola senza attesa), rulew (regola con attesa), rand (casuale), ppo, mz (MuZero)
REGOLA = "ft_sjf_th80"      # regola scritta di riferimento: ordine (fifo|sjf|lpt) + livello (L1|L07|th70|th80|th90); la versione con attesa e' ftW_...
CASI_N_SEED = 60            # seed per ciascuno scenario nella modalita' "scenari" (7000 .. 7000+N-1)

# ------------------------------------------------------------------ modelli salvati (percorsi relativi alla cartella FT0)
PPO_CHECKPOINT = "modelli/ppo_ft0_best_r0"                   # senza .zip; accanto c'e' _obsrms.pkl
MUZERO_CHECKPOINT = "modelli/muzero_ft0_it56250.pth.tar"     # classic MuZero, iteration 56250
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
