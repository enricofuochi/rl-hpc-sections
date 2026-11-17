#!/bin/bash
# MuZero (semplice, deterministico) da zero su FT1, con REWARD DI SCARTO dalla regola di riferimento di FT1 (ft1lib/config_ft1.py: REGOLA).
# STESSA configurazione della run lunga di FT0: 500000 passi, batch 512, 45 aggiornamenti per raccolta di 8 episodi, td 10, buffer 200000 (replay uniforme), 100 simulazioni,
# lr 1e-3 con decadimento (x1 fino a 25000 iterazioni, x0.1 fino a 50000, poi x0.01), consistenza 2, LayerNorm, reti ausiliarie ridotte (MZ_SLIM=1), reward di scarto scala 100.
# Uso (da qualsiasi cartella):   caffeinate -i bash FT1/train_muzero.sh <nome_run> [seed=0] [smoke]
#   smoke = prova rapidissima (verifica che tutto giri: raccolta, addestramento, checkpoint) -> usa un nome come ft1_smoke
# Le run vanno in FT1/runs/<nome_run>/ ; log in train.log.
FT1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$FT1/.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT:$ROOT/muzero:$FT1"; export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MZ_NORM="${MZ_NORM:-LN}" MZ_SLIM="${MZ_SLIM:-1}"
NAME="${1:?uso: train_muzero.sh <nome_run> [seed] [smoke]}"; SEED="${2:-0}"; SMOKE="${3:-}"
REGOLA_DEFAULT="$(python -c "import sys; sys.path.insert(0,'$FT1'); from ft1lib.config_ft1 import REGOLA; print(REGOLA)")"
export MZ_RESID="${MZ_RESID-$REGOLA_DEFAULT}" MZ_RESID_SCALE="${MZ_RESID_SCALE:-100}" MZ_STOP=1e9
export MZ_SIMS="${MZ_SIMS:-100}" MZ_UPD="${MZ_UPD:-45}" MZ_BATCH="${MZ_BATCH:-512}" MZ_LR="${MZ_LR:-0.001}" MZ_TD="${MZ_TD:-10}" MZ_RB="${MZ_RB:-200000}" MZ_STEPS="${MZ_STEPS:-500000}" MZ_LR_DECAY="${MZ_LR_DECAY:-1}" MZ_LR_DECAY_ITERS="${MZ_LR_DECAY_ITERS:-50000}" MZ_CKPT_EVERY="${MZ_CKPT_EVERY:-1000}"
if [ -z "$SMOKE" ] && [ -e "$FT1/runs/$NAME/data_mz" ]; then echo "la run $NAME esiste gia' (non la sovrascrivo)"; exit 1; fi
mkdir -p "$FT1/runs/$NAME"; cd "$FT1/runs/$NAME"
echo "FT1 | tipo ${MZ_KIND:-muzero} | residuo '$MZ_RESID' scala $MZ_RESID_SCALE | sims $MZ_SIMS upd $MZ_UPD batch $MZ_BATCH lr $MZ_LR decay $MZ_LR_DECAY/$MZ_LR_DECAY_ITERS td $MZ_TD rb $MZ_RB slim $MZ_SLIM | passi $MZ_STEPS | seed $SEED $SMOKE" | tee -a train.log
python "$FT1/muzero/mz_config_ft1.py" "${MZ_KIND:-muzero}" "$SEED" $SMOKE 2>&1 | tee -a train.log
