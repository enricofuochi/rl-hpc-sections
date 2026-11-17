"""Reti ausiliarie della perdita di consistenza (EfficientZero-style) di LightZero, con dimensioni ridotte.
Default LightZero: proj_hid 1024, proj_out 1024, pred_hid 512, pred_out 1024 (~3.3M parametri su ~3.6M totali, latent 128).
SLIM: 256/256/128/256 (~0.58M parametri totali). Il latente principale (128), il peso della consistenza (2) e l'unroll (5) NON cambiano.
- MZ_SLIM=1 attiva la riduzione in fase di addestramento (letta da mz_config_ft.py; run_mm_lungo.py la applica direttamente, default 1).
- Gli VALUTATORI non dipendono da variabili d'ambiente: ricostruiscono le dimensioni leggendo le forme dei pesi dal checkpoint (dims_from_ckpt),
  quindi funzionano sia con i checkpoint vecchi (dimensioni di default) sia con quelli ridotti."""
import os
SLIM = dict(proj_hid=256, proj_out=256, pred_hid=128, pred_out=256)
DEFAULT = dict(proj_hid=1024, proj_out=1024, pred_hid=512, pred_out=1024)


def slim_enabled(default='0'):
    return os.environ.get('MZ_SLIM', default) == '1'


def dims_from_ckpt(path):
    """Dimensioni delle reti ausiliarie dedotte dal checkpoint; None se il checkpoint non ha le reti di consistenza."""
    import torch
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    def shp(suffix):
        for k, v in sd.items():
            if k.endswith(suffix): return tuple(v.shape)
        return None
    a, b, c, d = shp('projection.0.weight'), shp('projection.6.weight'), shp('prediction_head.0.weight'), shp('prediction_head.3.weight')
    if None in (a, b, c, d): return None
    dims = dict(proj_hid=a[0], proj_out=b[0], pred_hid=c[0], pred_out=d[0])
    assert b[1] == dims['proj_hid'] and c[1] == dims['proj_out'] and d[1] == dims['pred_hid'], (a, b, c, d)
    return dims


def n_aux_params(dims, latent=128):
    ph, po, qh, qo = dims['proj_hid'], dims['proj_out'], dims['pred_hid'], dims['pred_out']
    bn = lambda n: 2 * n
    return (latent * ph + ph + bn(ph)) + (ph * ph + ph + bn(ph)) + (ph * po + po + bn(po)) + (po * qh + qh + bn(qh)) + (qh * qo + qo)
