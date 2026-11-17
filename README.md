# RL for HPC power capping and scheduling: FT0 / FT1

Code for the fast/efficient-section cluster environments **FT0** and **FT1** of the thesis, with the four decision-makers compared there:
a hand-written rule, a privileged one/two-step planner (PLAN), MaskablePPO and MuZero.

* **FT0**: 24 jobs in the queue at t=0, a cluster of 6 fast (1000 W) + 10 efficient (600 W) nodes, tick of 600 s, power level L chosen independently at each tick in {1.0, 0.7}.
* **FT1**: FT0 with the power level as a *state* that moves by at most 0.1 per tick (L-, L=, L+; L in [0.5, 1.0]); 15 actions, 146-number observation.
  Cost per tick: `(n_sys * dt_h + w * L * dt_h) / N` with `w = 5.7`.

## Layout

```
hpcpower/      FT environment (env_ft.py, env_ft0.py), FT0 rules, residual-reward shaping (resid.py), MaskablePPO utilities (ppo.py), LightZero wrapper for FT0
muzero/        LightZero configs/evaluation for FT0 (mz_config_ft.py, mz_eval_ft.py) and slim_aux.py (reduced auxiliary heads of the consistency loss)
FT0/           01_esegui_politiche, 03_visualizzatore, 04_tabelle, lib/, config/, modelli/ (checkpoints, not tracked)
FT1/           numbered scripts (see below), play_episode.py, train_ppo.py, train_muzero*.sh, ft1lib/ (environment + rule + planner + LightZero wrapper), lib/, muzero/, config/
```

`FT1/config/config_fissa.py` holds the configuration used for every result of the thesis (do not edit); `config_utente.py` is a free copy for experiments.
Seeds: calibration 900-929, validation 1000-1029, **fresh test 9000-9099** for FT1 (FT0 test 7000-7099), training seeds >= 10000.

## Install

```
pip install -r requirements.txt            # environments, rules, PLAN, PPO, figures
pip install -r requirements-muzero.txt     # additionally, to run or train MuZero (torch + LightZero 0.2.0)
```

Tested with Python 3.11 (numpy 1.26, gymnasium 0.28, stable-baselines3 / sb3-contrib 2.9). Rules, PLAN and PPO also run on newer numpy/gymnasium.
Checkpoints are expected in `FT1/modelli/` (`ppo_ft1_best_r0.zip`, `ppo_ft1_best_r0_obsrms.pkl`, `muzero_ft1_it56880.pth.tar`) and `FT0/modelli/` (same naming with `ft0`, MuZero iteration 56250).

## Quick start: play one episode

```
python FT1/play_episode.py --seed 9000 --players rule ppo plan1      # costs of each player on the same instance
python FT1/play_episode.py --seed 9000 --players rule ppo mz --log   # + decision-by-decision log (mz needs LightZero)
python FT1/play_episode.py --seed 9000 --players rule ppo mz --html episode.html   # side-by-side interactive replay
```

The episode is fully determined by the seed, so all players face the same instance. `--html` writes a single self-contained page: three clusters side by side,
power level and requested/delivered power, and a decision inspector showing what PPO (probabilities) and MuZero (MCTS visits) would do in each state.

## Reproducing the numbers (no training needed)

| | rule | PLAN 1 step | PPO |
|---|---|---|---|
| FT1, test 9000-9099 (mean cost, h) | 2.3061 | 1.8530 | 1.8758 |
| FT0, test 7000-7099 (mean cost, h) | 2.3838 | - | 1.9357 |

`python FT1/reproduce_results.py` (and `python FT0/reproduce_results.py`) re-plays the rule, PLAN and PPO on the fresh test seeds and checks the means above to 4 decimals (add `--players ... mz` for MuZero, which needs LightZero and is not exactly repeatable).
`python FT1/00_verifica.py` runs the environment checks (shapes, masks, exact equivalence with FT0 for delta = 0.3 / L_min = 0.7, residual-reward identity).
`python FT1/02_esegui_politiche.py` (interactive) plays the players on many seeds and writes `FT1/output/<experiment>/tracce.pkl`, which the figure, table and viewer scripts read.

## Training

```
python FT1/train_ppo.py [seed] [steps]                      # MaskablePPO, preset of the thesis (300000 steps)
caffeinate -i bash FT1/train_muzero_lungo.sh ft1_lungo_s0 0  # MuZero with periodic validation (56880 iterations in the thesis)
```

MuZero is classic LightZero MuZero (`lzero.policy.muzero`): MLP, latent 128, LayerNorm, SimNorm, 601-class reward/value supports, consistency loss, residual reward
(potential-based shaping from the reference rule, scale 100). All hyper-parameters are in `FT1/muzero/mz_config_ft1.py` and `FT1/train_muzero.sh`.

## FT1 scripts

| script | role |
|---|---|
| `00_verifica.py` | environment checks (shapes, masks, exact equivalence with FT0, residual-reward identity) |
| `01_baseline.py`, `02_esegui_politiche.py` | **core**: rules / PLAN / PPO baselines; plays all players on many seeds and writes `output/<experiment>/tracce.pkl` (read by the scripts below) |
| `04_visualizzatore.py`, `play_episode.py` | interactive HTML replay of recorded traces / of a single episode |
| `05_tabelle.py`, `12_scomposizione_costo.py` | comparison tables (cost, general metrics) and cost decomposition (+ csv) |
| `09_ricerca_vs_rete.py` | MuZero with the network only vs. with 5/25/100 simulations (also the helper module of script 10) |

Figures of the thesis (written to `output/<experiment>/figures/` or `output/`):

| script | figure |
|---|---|
| `10_figura_ricerca.py` | search vs. network-only along training (`figure_ricerca_<run>`) |
| `11_curve_addestramento.py` | training curves (`curve_addestramento_<run>`) |
| `16_figura_valore_fiducia.py` | value bias, top-action probability, policy entropy (`valore_e_fiducia`) |
| `14_figure_finali.py` | Gantt of the same episode for the three players (`gantt_triplo_id_seed9097`) and energy dashboard (`cruscotto_energia_2p`) |
| `13_episodio_due_pannelli.py` | example episode in two panels (`episodio_due_pannelli`) |
