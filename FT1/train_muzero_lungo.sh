#!/bin/bash
# Run LUNGA su FT1 con valutazione automatica durante l'addestramento (stesso protocollo di FT0): ogni 10000 iterazioni valuta il checkpoint su validazione 1000-1029 (2 repliche);
# alla fine l'ULTIMO checkpoint con 3 repliche. Riassunto: FT1/runs/<nome_run>/eval_ft1_val_riassunto.txt ; log: runs/<nome_run>.out e .osserva.out
# Uso:   caffeinate -i bash FT1/train_muzero_lungo.sh <nome_run> [seed=0]
FT1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$FT1/.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT:$ROOT/muzero:$FT1"; export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
RUN="${1:?nome run}"; SEED="${2:-0}"; mkdir -p "$FT1/runs"
bash "$FT1/train_muzero.sh" "$RUN" "$SEED" > "$FT1/runs/$RUN.out" 2>&1 &
PID=$!
echo "addestramento avviato (pid $PID): $RUN"
N=10000
while true; do
  CK=$(ls "$FT1"/runs/$RUN/data_mz/*/ckpt/iteration_${N}.pth.tar 2>/dev/null | head -1)
  if [ -n "$CK" ]; then
    sleep 20
    MZ_SIMS=100 python "$FT1/muzero/mz_eval_ft1.py" "$RUN" muzero $N 2 val >> "$FT1/runs/$RUN.osserva.out" 2>&1
    tail -n 3 "$FT1/runs/$RUN/eval_ft1_val_riassunto.txt"
    N=$((N+10000))
  elif ! kill -0 $PID 2>/dev/null; then
    break
  else
    sleep 120
  fi
done
MZ_SIMS=100 python "$FT1/muzero/mz_eval_ft1.py" "$RUN" muzero last 3 val >> "$FT1/runs/$RUN.osserva.out" 2>&1
cat "$FT1/runs/$RUN/eval_ft1_val_riassunto.txt"
echo "FINITO $RUN"
