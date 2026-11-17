"""make_env per FT1: env='ft1' -> FT1Env, altrimenti la fabbrica di hpcpower."""


def make_env(kw):
    if kw.get("env") == "ft1":
        from .env_ft1 import FT1Env
        return FT1Env(**{k: v for k, v in kw.items() if k != "env"})
    from hpcpower.factory import make_env as _m
    return _m(kw)
