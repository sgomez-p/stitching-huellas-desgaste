#!/usr/bin/env python3
"""Genera datos SINTETICOS para probar los scripts sin medidas reales.

  * ejemplos/campos/   4 campos .gsf solapados de una huella de desgaste
                       (para coser_huella.py)
  * ejemplos/ensayo_sintetico.tst   ensayo reciprocante de ~10 min a 100 Hz
                       (para exportar_origin_bof.py / bof_core.py)

Uso:  python ejemplos/generar_datos_sinteticos.py
"""
import os
import sys

import numpy as np
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
from coser_huella import guardar_gsf  # noqa: E402

rng = np.random.default_rng(0)


def campos_huella(n_campos=4, alto=300, ancho=400, solape=0.3, px=1.5):
    """Superficie rugosa con una huella horizontal, cortada en campos solapados
    y con un offset Z distinto en cada campo (como al mover la platina a mano)."""
    paso = int(ancho * (1 - solape))
    W = paso * (n_campos - 1) + ancho
    rug = ndimage.gaussian_filter(rng.normal(size=(alto, W)), 2) * 0.4
    y = np.arange(alto)[:, None]
    huella = -3.0 * np.exp(-((y - alto / 2) / (alto * 0.15)) ** 2)   # surco de 3 um
    sup = rug + huella + 0.002 * np.arange(W)[None, :]                # ligera inclinacion
    os.makedirs(os.path.join(AQUI, "campos"), exist_ok=True)
    for k in range(n_campos):
        x0 = k * paso + int(rng.integers(-3, 4))
        x0 = min(max(x0, 0), W - ancho)
        c = sup[:, x0:x0 + ancho] + rng.normal(0, 0.5)               # offset Z aleatorio
        guardar_gsf(os.path.join(AQUI, "campos", f"MUESTRA-A-2N-01-{k + 1:02d}.gsf"), c, px)
    print(f"  {n_campos} campos en {os.path.join(AQUI, 'campos')}")


def tst_sintetico(minutos=10, hz=100, fz=2.0, cof=0.45):
    """Fichero .tst con la estructura que lee bof_core.leer_tst:
    cabecera de relleno + registros de 33 B (Fx, Fz, Z1 f64; contador 2*i32; 0x24)."""
    n = int(minutos * 60 * hz)
    t = np.arange(n) / hz
    signo = np.sign(np.sin(2 * np.pi * 5 * t))                         # reciprocante 5 Hz
    mu = cof - 0.15 * np.exp(-t / 60) + rng.normal(0, 0.02, n)        # rodaje + ruido
    Fz = fz + rng.normal(0, 0.02, n)
    Fx = signo * mu * Fz
    Z1 = -2e-5 * t + rng.normal(0, 2e-4, n)
    cont = (t * 10_000).astype(np.int64)
    dt = np.dtype([("Fx", "<f8"), ("Fz", "<f8"), ("Z1", "<f8"),
                   ("lo", "<i4"), ("hi", "<i4"), ("sep", "u1")])
    rec = np.zeros(n, dtype=dt)
    rec["Fx"], rec["Fz"], rec["Z1"] = Fx, -Fz, Z1
    rec["lo"] = (cont & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    rec["hi"] = (cont >> 32).astype(np.int32)
    rec["sep"] = 0x24
    ruta = os.path.join(AQUI, "ensayo_sintetico.tst")
    with open(ruta, "wb") as f:
        f.write(b"\x00" * 1200)
        f.write(rec.tobytes())
    print(f"  ensayo .tst en {ruta}")


if __name__ == "__main__":
    campos_huella()
    tst_sintetico()
