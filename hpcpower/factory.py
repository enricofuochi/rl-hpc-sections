"""Environment factory for FT0 (env="ft0"). FT1 has its own factory (FT1/ft1lib/factory_ft1.py) that falls back to this one."""


def make_env(env_kw):
    kind = env_kw.get("env")
    if kind == "ft0":
        from .env_ft0 import FT0Env
        return FT0Env(**{k: v for k, v in env_kw.items() if k != "env"})
    raise ValueError(f"unknown environment kind: {kind!r}")
