"""
Configurazione LightZero per FT0 (24 job a t=0, tick 600 s, ~51 decisioni, deterministico, osservazione completa 146 numeri, 14 azioni).
Copia delle chiavi di mz_config.py (gia' funzionanti con LightZero 0.2.0), con ambiente e parametri propri. Additivo: mz_config.py non e' toccato.
Uso (dalla radice del progetto, con PYTHONPATH=radice):
    python muzero/mz_config_ft.py muzero [seed]          # MuZero semplice (ambiente deterministico: nessun ramo di chance)
    ... smoke                                            # prova rapidissima
Variabili (default della run lunga): MZ_SIMS (100) MZ_UPD (45) MZ_BATCH (512) MZ_LR (0.001) MZ_TD (10) MZ_RB (200000) MZ_STEPS (500000) MZ_CKPT_EVERY (1000) MZ_NORM (LN)
           MZ_LR_DECAY (1: lr x1 fino a meta', x0.1 fino a MZ_LR_DECAY_ITERS, poi x0.01) MZ_LR_DECAY_ITERS (50000, circa 0.9 x le iterazioni totali)
           MZ_RESID (ft_sjf_th80) MZ_RESID_SCALE (100) [lette dal wrapper hpcpower/lz_env_ft.py] MZ_INIT (checkpoint da cui ripartire).
"""
import os, sys
from easydict import EasyDict

FT0_KW = dict(env="ft0", n_jobs=24, dt=600.0, w=5.7, xobs=True, qk=24)     # w tarato a priori su semi 900-929 (regola sjf, L=1: energia = flusso)
NORM = os.environ.get('MZ_NORM', 'LN')


def make_cfg_ft(kind="muzero", smoke=False, seed=0, tag="", model_path=None):
    from hpcpower.factory import make_env
    kw = dict(FT0_KW)
    probe = make_env(kw)
    A, D = probe.n_actions, probe.obs_dim                                   # 14 azioni, 146 numeri
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
        exp_name=f"data_mz/ft0_{kind}_ns{num_simulations}_seed{seed}" + tag + ("_smoke" if smoke else ""),
        env=dict(
            env_id='hpc_ft', hpc_kw=kw, stop_value=float(os.environ.get('MZ_STOP', 1e9)),
            collector_env_num=collector_env_num, evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num, manager=dict(shared_memory=False),
        ),
        policy=policy,
    )
    cfg["env"]["train_seeds"] = None
    cfg["env"]["eval_seeds"] = None
    create = dict(
        env=dict(type='hpc_ft_lightzero', import_names=['hpcpower.lz_env_ft']),
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
