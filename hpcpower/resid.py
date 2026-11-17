"""Reward di SCARTO dalla regola (shaping a potenziale, additivo; spento se MZ_RESID non e' impostata).

Phi(s) = - costo (ore) che la regola di riferimento sosterrebbe da s fino alla fine dell'episodio, simulato su una COPIA dell'ambiente.
Reward modificato: r' = r + Phi(s') - Phi(s)  (Phi(terminale) = 0). La somma telescopica vale  costo_regola(s0) - costo_agente  (sconto 1.0):
la politica ottima e' identica a quella del reward originale (Ng, Harada, Russell 1999), ma il valore che la rete deve apprendere e' il guadagno rispetto alla regola
(0 per la regola stessa) e non il costo totale (~3 ore). Il simulatore con i tipi veri serve solo per calcolare il riferimento in addestramento; l'agente osserva sempre e solo l'osservazione onesta.
Il reward e' moltiplicato per MZ_RESID_SCALE (default 10: decimi di ora, come SEC0/BAT)."""
import copy


def costo_residuo_h(env, regola):
    """Costo (ore) della regola da lo stato corrente a fine episodio, su una copia dell'ambiente (non modifica env)."""
    e = copy.deepcopy(env); T = 0.0
    while True:
        _, r, te, tr, _ = e.step(int(regola(e))); T -= r
        if te or tr: return T / 3600.0
