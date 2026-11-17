"""Wrapper LightZero per FT1 (ft1lib/env_ft1.py). Copia di hpcpower/lz_env_ft.py con la fabbrica e le regole di FT1. Additivo: nessun file di hpcpower/ e' toccato.

Interfaccia LightZero: reset() -> dict {'observation', 'action_mask', 'to_play': -1, 'timestep'}; step(a) -> BaseEnvTimestep(obs_dict, reward, done, info).
Reward di FTEnv = -costo in "secondi equivalenti" (ore x 3600). Reward = r / 3600 (ore). Se MZ_RESID=<nome regola in ft1lib.policies_ft1.FT1_POLICIES> il reward e' di SCARTO dalla regola
(shaping a potenziale, come hpcpower/resid.py): r' = scala * (r + Phi(s') - Phi(s)), Phi = -costo (ore) della regola da quello stato; Phi(terminale)=0. Scala: MZ_RESID_SCALE (default 100).
La somma dei r' = scala * (costo regola - costo agente); con l'azione della regola r' = 0.
Seed: raccolta con seed casuali in [10000, 1e6), mai validazione 1000-1029, calibrazione 900-929, sviluppo 3000-3099, test 5000-5099.
"""
import copy
import os
from typing import Dict, Union

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from hpcpower.resid import costo_residuo_h

from .config_ft1 import KW as FT1_KW
from .factory_ft1 import make_env
from .policies_ft1 import FT1_POLICIES

REWARD_UNIT = 3600.0
TRAIN_SEED_LOW, TRAIN_SEED_HIGH = 10_000, 1_000_000


@ENV_REGISTRY.register('hpc_ft1_lightzero')
class HPCFT1LZEnv(BaseEnv):
    config = dict(
        env_id='hpc_ft1',
        hpc_kw=dict(FT1_KW),
        eval_seeds=None,
        train_seeds=None,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        kw = dict(cfg.get('hpc_kw', None) or self.config['hpc_kw'])
        self._kw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in kw.items()}
        self._env = make_env(self._kw)
        self._eval_seeds = list(cfg.get('eval_seeds') or [])
        self._train_seeds = [int(x) for x in (cfg.get('train_seeds') or [])]
        self._eval_i = 0
        self._rng = np.random.RandomState(0)
        n = self._env.n_actions
        self._action_space = gym.spaces.Discrete(n)
        self._observation_space = gym.spaces.Box(0.0, 1.0, shape=(self._env.obs_dim,), dtype=np.float32)
        self._reward_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self._timestep = 0
        self._eval_episode_return = 0.0
        self._resid = None
        self._rscale, self._phi = 1.0, 0.0
        if os.environ.get('MZ_RESID'):
            self._resid = FT1_POLICIES[os.environ['MZ_RESID']]
            self._rscale = float(os.environ.get('MZ_RESID_SCALE', 100.0))

    def _obs(self, mask=None) -> Dict[str, np.ndarray]:
        mask = self._env.action_masks() if mask is None else mask
        return {'observation': self._env._build_observation().astype(np.float32), 'action_mask': mask.astype(np.int8),
                'to_play': -1, 'timestep': self._timestep}

    def reset(self) -> Dict[str, np.ndarray]:
        if self._eval_seeds:
            seed = self._eval_seeds[self._eval_i % len(self._eval_seeds)]
            self._eval_i += 1
        elif self._train_seeds:
            seed = int(self._rng.choice(self._train_seeds))
        else:
            seed = int(self._rng.randint(TRAIN_SEED_LOW, TRAIN_SEED_HIGH))
        self._env.reset(seed=seed)
        self._timestep, self._eval_episode_return = 0, 0.0
        if self._resid is not None:
            self._phi = -costo_residuo_h(self._env, self._resid)
        return self._obs()

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray):
            action = int(action.reshape(-1)[0])
        _, r, terminated, truncated, info = self._env.step(int(action))
        rew = float(r / REWARD_UNIT)
        done = bool(terminated or truncated)
        if self._resid is not None:
            phi1 = 0.0 if done else -costo_residuo_h(self._env, self._resid)
            rew = self._rscale * (rew + phi1 - self._phi); self._phi = phi1
        self._eval_episode_return += rew
        self._timestep += 1
        out = {'makespan_h': float(info.get('makespan', 0.0)) / 3600.0, 'invalid_action': False}
        if done:
            out['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(self._obs(info['action_mask']), rew, done, out)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def close(self) -> None:
        pass

    def random_action(self) -> np.ndarray:
        valid = np.flatnonzero(self._env.action_masks())
        return np.array([int(self._rng.choice(valid))], dtype=np.int64)

    @property
    def legal_actions(self):
        return np.flatnonzero(self._env.action_masks())

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return f"HPCFT1LZEnv(n_actions={self._action_space.n}, obs_dim={self._observation_space.shape[0]})"
