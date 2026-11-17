"""
Configurazione LightZero per FT1 (FT0 con livello di potenza L come stato e mosse L-/L=/L+; 15 azioni, osservazione 146 numeri). Copia di muzero/mz_config_ft.py: STESSI iperparametri di FT0, cambiano solo ambiente, wrapper e nome.
Uso (da FT1/, con PYTHONPATH=radice:FT1, lo imposta gia' train_muzero.sh):
    python muzero/mz_config_ft1.py muzero [seed]          # MuZero semplice (ambiente deterministico: nessun ramo di chance)
    ... smoke                                            # prova rapidissima
Variabili (default della run lunga): MZ_SIMS (100) MZ_UPD (45) MZ_BATCH (512) MZ_LR (0.001) MZ_TD (10) MZ_RB (200000) MZ_STEPS (500000) MZ_CKPT_EVERY (1000) MZ_NORM (LN)
           MZ_LR_DECAY (1: lr x1 fino a meta', x0.1 fino a MZ_LR_DECAY_ITERS, poi x0.01) MZ_LR_DECAY_ITERS (50000, circa 0.9 x le iterazioni totali)
           MZ_RESID (ft1lib.config_ft1.REGOLA) MZ_RESID_SCALE (100) [lette dal wrapper ft1lib/lz_env_ft1.py] MZ_INIT (checkpoint da cui ripartire).
"""
import os, sys
from pathlib import Path
from easydict import EasyDict
_FT1 = str(Path(__file__).resolve().parent.parent)
if _FT1 not in sys.path: sys.path.insert(0, _FT1)
from ft1lib.config_ft1 import KW as FT0_KW      # (il nome FT0_KW resta per compatibilita' con mz_eval) parametri dell'ambiente FT1; w=5.7 come FT0
NORM = os.environ.get('MZ_NORM', 'LN')


def make_cfg_ft(kind="muzero", smoke=False, seed=0, tag="", model_path=None):
    from ft1lib.factory_ft1 import make_env
    kw = dict(FT0_KW)
    probe = make_env(kw)
    A, D = probe.n_actions, probe.obs_dim                                   # 15 azioni, 146 numeri
    collector_env_num, n_episode = (2, 2) if smoke else (8, 8)
    evaluator_env_num = 1 if smoke else 4
    num_simulations = 8 if smoke else int(os.environ.get("MZ_SIMS", 100))
    update_per_collect = 5 if smoke else int(os.environ.get("MZ_UPD", 45))
    batch_size = 32 if smoke else int(os.environ.get("MZ_BATCH", 512))
    model = dict(
        observation_shape=D, action_space_size=A, model_type='mlp', latent_state_dim=128,
        self_supervised_learning_loss=True, discrete_action_encoding_type='one_hot', norm_type=NORM,
        reward_support_range=(-30., 31., 1.), value_support_range=(-30., 31., 1.),   # NOTA: in questo LightZero i campi range sono IGNORATI; il supporto effettivo e' support_scale=300 -> 601 classi per reward e valore
    )
    model['lstm_hidden_size'] = 128
    from slim_aux import SLIM, slim_enabled                                  # MZ_SLIM=1: reti ausiliarie di consistenza ridotte (256/256/128/256); latente 128, peso 2, unroll 5 invariati
    if slim_enabled(): model.update(SLIM)
    policy = dict(
        model=model, model_path=model_path, cuda=False, env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=20000,
        td_steps=10 if smoke else int(os.environ.get('MZ_TD', 10)),
        num_unroll_steps=5, discount_factor=1.0,
        update_per_collect=update_per_collect, batch_size=batch_size, optim_type='Adam',
        piecewise_decay_lr_scheduler=(not smoke) and bool(int(os.environ.get('MZ_LR_DECAY', 1))), threshold_training_steps_for_final_lr=int(os.environ.get('MZ_LR_DECAY_ITERS', 50000)), learning_rate=float(os.environ.get('MZ_LR', 0.001)),
        ssl_loss_weight=2, use_priority=False,                      # replay uniforme, esplicito (classico: default False)
        num_simulations=num_simulations, reanalyze_ratio=0., n_episode=n_episode,
        eval_freq=20 if smoke else 4000, replay_buffer_size=int(float(os.environ.get('MZ_RB', 200000))),
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=20 if smoke else int(os.environ.get("MZ_CKPT_EVERY", 1000))))),
        collector_env_num=collector_env_num, evaluator_env_num=evaluator_env_num,
    )
    cfg = dict(
        exp_name=f"data_mz/ft1_{kind}_ns{num_simulations}_seed{seed}" + tag + ("_smoke" if smoke else ""),
        env=dict(
            env_id='hpc_ft1', hpc_kw=kw, stop_value=float(os.environ.get('MZ_STOP', 1e9)),
            collector_env_num=collector_env_num, evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num, manager=dict(shared_memory=False),
        ),
        policy=policy,
    )
    cfg["env"]["train_seeds"] = None
    cfg["env"]["eval_seeds"] = None
    create = dict(
        env=dict(type='hpc_ft1_lightzero', import_names=['ft1lib.lz_env_ft1']),
        env_manager=dict(type='base' if smoke else 'subprocess'),
        policy=dict(type='muzero', import_names=['lzero.policy.muzero']),
    )
    return EasyDict(cfg), EasyDict(create)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    kind = args[0] if args else "muzero"
    smoke = "smoke" in sys.argv
    seed = int(args[1]) if len(args) > 1 else 0
    init = os.environ.get("MZ_INIT") or None
    from lzero.entry import train_muzero
    main_cfg, create_cfg = make_cfg_ft(kind, smoke, seed=seed, model_path=init)
    train_muzero([main_cfg, create_cfg], seed=seed, model_path=main_cfg.policy.model_path,
                 max_env_step=3000 if smoke else int(os.environ.get("MZ_STEPS", 500000)))
