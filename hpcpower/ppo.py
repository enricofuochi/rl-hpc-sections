"""MaskablePPO utilities for FT0/FT1 (sb3-contrib): training wrapper, agent with inference helpers, paired evaluation.

The training protocol is the one used in the thesis (preset "mia"): MLP [64, 64], lr 3e-4, gamma 0.999, lambda 0.95, entropy 1e-2, 8 environments x 512 steps,
batch 256, no observation normalisation, raw reward divided by `reward_scale` (1500 for FT0/FT1). Every `chunk` steps the agent is evaluated on the validation seeds
(deterministic arg-max, cost difference paired with the reference rule); the returned agent is the checkpoint with the best 3-point moving average of that difference.
"""
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym

PRESET_MIA = dict(n_envs=8, reward_scale=36000.0,
                  ppo=dict(n_steps=512, batch_size=256, gamma=0.999, gae_lambda=0.95, learning_rate=3e-4, ent_coef=1e-2))
VAL_SEEDS_B = range(2000, 2030)      # second validation set, only used to report the chosen checkpoint


class TrainEnv(gym.Wrapper):
    """Environment with the reward divided by `reward_scale` and `action_masks()` exposed to sb3-contrib."""
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        o, r, te, tr, info = self.env.step(action)
        return o, r / self.reward_scale, te, tr, info

    def action_masks(self):
        return self.env.unwrapped.action_masks()

    def get_wrapper_attr(self, name):          # gymnasium < 0.29 lacks it; stable-baselines3 needs it (get_attr / env_method)
        return getattr(self, name)


class SeedPool(gym.Wrapper):
    """Every reset draws a seed at random from `seeds` (with a single seed the trace is always the same)."""
    def __init__(self, env, seeds, rs):
        super().__init__(env)
        self.seeds, self.rs = list(seeds), np.random.RandomState(rs)

    def reset(self, **kw):
        kw["seed"] = int(self.rs.choice(self.seeds))
        return self.env.reset(**kw)

    def get_wrapper_attr(self, name):
        return getattr(self, name)


class Agent:
    """MaskablePPO model plus the observation-normalisation statistics (None when the preset does not normalise)."""
    def __init__(self, model, obs_rms=None, clip=10.0):
        self.model, self.obs_rms, self.clip = model, obs_rms, clip

    def act(self, env, deterministic=True):
        o = env._build_observation()
        if self.obs_rms is not None:
            o = np.clip((o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)
        a, _ = self.model.predict(o, action_masks=env.action_masks(), deterministic=deterministic)
        return int(a)

    def save(self, path):
        self.model.save(path)
        with open(path + "_obsrms.pkl", "wb") as f:
            pickle.dump(self.obs_rms, f)

    @staticmethod
    def load(path):
        from sb3_contrib import MaskablePPO
        with open(path + "_obsrms.pkl", "rb") as f:
            rms = pickle.load(f)
        return Agent(MaskablePPO.load(path), rms)


def make_agent_policy(agent, deterministic=True):
    return lambda env: agent.act(env, deterministic)


def _episode_cost_h(policy_fn, seed, make_env, env_kw):
    env = make_env(env_kw)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    cost = 0.0
    while True:
        _, r, te, tr, _ = env.step(policy_fn(env))
        cost -= r
        if te or tr:
            return cost / 3600.0


def eval_agent(policy_fn, seeds, make_env, env_kw):
    """Episode cost (hours) for each seed. Saves and restores NumPy's global generator (reset(seed=...) reseeds it)."""
    state = np.random.get_state()
    try:
        return pd.Series({s: _episode_cost_h(policy_fn, s, make_env, env_kw) for s in seeds})
    finally:
        np.random.set_state(state)


def _stats(d):
    return d.mean(), d.std() / np.sqrt(len(d))


def train_agent(make_env, env_kw, reference, floor, val_seeds, val_seeds_b=VAL_SEEDS_B, run_seed=0, total_steps=300_000, chunk=25_000,
                reward_scale=1500.0, out_prefix="ppo", verbose=True):
    """Train one MaskablePPO run. `reference` and `floor` are policy functions env -> action (reference rule and random-legal floor).
    Returns (agent of the selected checkpoint, DataFrame with the validation history). Intermediate checkpoints are saved as <out_prefix>_r<seed>_<steps>."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    P = PRESET_MIA
    kw = dict(policy_kwargs=dict(net_arch=[64, 64]), **P["ppo"])
    venv = DummyVecEnv([(lambda: TrainEnv(make_env(env_kw), reward_scale)) for _ in range(P["n_envs"])])
    model = MaskablePPO("MlpPolicy", venv, verbose=0, seed=5000 + run_seed, **kw)      # seed outside validation and test ranges

    ref_a, floor_a = eval_agent(reference, val_seeds, make_env, env_kw), eval_agent(floor, val_seeds, make_env, env_kw)
    if verbose:
        print(f"run {run_seed} | validation cost (h): reference {ref_a.mean():.2f} | random {floor_a.mean():.2f}", flush=True)
    hist, paths = [], []
    for i in range(total_steps // chunk):
        model.learn(chunk, reset_num_timesteps=False)
        agent = Agent(model, None)
        mk = eval_agent(make_agent_policy(agent), val_seeds, make_env, env_kw)
        d = mk - ref_a
        path = f"{out_prefix}_r{run_seed}_{(i + 1) * chunk}"
        agent.save(path)
        paths.append(path)
        hist.append((mk.mean(), *_stats(d), (mk - floor_a).mean()))
        if verbose:
            print(f"run {run_seed} | {(i + 1) * chunk:>8} steps | validation cost {hist[-1][0]:6.2f} h | vs reference {hist[-1][1]:+6.2f} ± {hist[-1][2]:4.2f} h | "
                  f"vs random {hist[-1][3]:+6.2f} h", flush=True)

    m = np.array([h[1] for h in hist])
    smooth = np.array([m[max(0, i - 1):i + 2].mean() for i in range(len(m))])      # 3-point moving average
    best = int(smooth.argmin())
    agent = Agent.load(paths[best])
    if verbose:
        ref_b = eval_agent(reference, val_seeds_b, make_env, env_kw)
        dB = eval_agent(make_agent_policy(agent), val_seeds_b, make_env, env_kw) - ref_b
        print(f"run {run_seed} | selected checkpoint: {paths[best]} (moving average {smooth[best]:+.2f} h vs reference) | "
              f"validation B: {dB.mean():+.2f} ± {dB.std() / np.sqrt(len(dB)):.2f} h", flush=True)
    return agent, pd.DataFrame(hist, columns=["cost_h", "vs_reference_h", "se_h", "vs_random_h"], index=[(i + 1) * chunk for i in range(len(hist))])
