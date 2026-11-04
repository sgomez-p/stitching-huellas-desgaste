#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bof_core.py — Lector de ficheros .tst y motor de calculo para ensayos
tribologicos reciprocantes (block-on-flat / pin-on-flat).

  leer_tst(ruta)   -> DataFrame con tiempo_s, Fz_N, Fx_N, Z1_mm, CoF
  calcular(df, P)  -> (df con columnas derivadas, stats, df diezmado)

P es un dict con al menos: id_ensayo, carga_N, velocidad_ms, frecuencia_hz,
ventana_s, pct_estacionario (y opcionalmente lote, tratamiento, zona).

Solo depende de numpy y pandas. Multiplataforma.
"""

import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_DEFAULTS = dict(
    lote             = "",
    tratamiento      = "",
    zona             = "",
    carga_N          = 2.0,
    velocidad_ms     = 0.30,
    frecuencia_hz    = 5.0,
    ventana_s        = 120.0,  # ventana de suavizado en SEGUNDOS (independiente del muestreo)
    pct_estacionario = 0.8,
)


# ═════════════════════════════════════════════════════════════════════════════
# PARSER BINARIO .TST — CETR/UMT
# Formato: 33 bytes/registro
#   Fx (f64,8b) + Fz (f64,8b) + Z1 (f64,8b)
#   + contador (2×i32,8b) + separador 0x24 (1b)
#   Tiempo = contador / 10 000 s
# ═════════════════════════════════════════════════════════════════════════════

def leer_tst(filepath: str) -> pd.DataFrame:
    data     = Path(filepath).read_bytes()
    n_bytes  = len(data)
    rec_size = 33
    sep_byte = 0x24

    # Fz mínimo adaptado a cargas bajas de ensayos BOF (ej. 2 N)
    FZ_MIN = 0.05
    FZ_MAX = 500.0

    dtype = np.dtype([
        ("Fx", "<f8"), ("Fz", "<f8"), ("Z1", "<f8"),
        ("lo", "<i4"), ("hi", "<i4"), ("sep", "u1"),
    ])

    def _run_length(start: int) -> int:
        """Nº de registros consecutivos válidos desde 'start', vectorizado
        (equivale al bucle while original, pero sin overhead de Python
        por registro: un struct.unpack_from por muestra era el cuello de
        botella real en archivos .tst grandes)."""
        max_recs = (n_bytes - start) // rec_size
        if max_recs <= 0:
            return 0
        arr = np.frombuffer(data, dtype=dtype, count=max_recs, offset=start)
        valid = (arr["sep"] == sep_byte) & np.isfinite(arr["Fz"])
        bad = np.flatnonzero(~valid)
        return int(bad[0]) if bad.size else max_recs

    best_start, best_count = None, 0
    i = 1000
    while i + rec_size <= n_bytes:
        if data[i + 32] == sep_byte:
            fz = struct.unpack_from("<d", data, i + 8)[0]
            if math.isfinite(fz) and FZ_MIN < abs(fz) < FZ_MAX:
                count = _run_length(i)
                if count > best_count:
                    best_count, best_start = count, i
                i += count * rec_size   # == j del bucle original (count ≥ 1 aquí)
                continue
        i += 1

    if best_start is None or best_count < 50:
        raise ValueError(
            f"No se encontró un bloque de datos válido.\n  Archivo: {filepath}"
        )

    print(f"  Parser: byte {best_start:,}, {best_count:,} registros")

    arr = np.frombuffer(data, dtype=dtype, count=best_count, offset=best_start)

    lo = arr["lo"].astype(np.int64)
    hi = arr["hi"].astype(np.int64)
    lo = np.where(lo < 0, lo + (1 << 32), lo)
    hi = np.where(hi < 0, hi + (1 << 32), hi)
    t  = (hi * (1 << 32) + lo) / 10_000.0
    t  = t - t[0]

    Fz  = np.abs(arr["Fz"])   # siempre positivo: carga normal es magnitud
    Fx  = arr["Fx"]           # Fx conserva signo (alternante ±)
    CoF = np.where(Fz > 0.05, np.abs(Fx / Fz), 0.0)

    return pd.DataFrame({
        "tiempo_s":  t,
        "Fz_N":      Fz,
        "Fx_N":      Fx,
        "Z1_mm":     arr["Z1"],
        "CoF":       CoF,
    })


def calcular(df: pd.DataFrame, P: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Núcleo de cálculo (sin GUI ni exportación). Recibe el df de leer_tst y
    los parámetros P (con id_ensayo); añade las columnas derivadas y devuelve
    (df, stats, df_p diezmado). Añade P['stroke_mm'].
    Reutilizable desde otros scripts."""
    df["distancia_m"] = df["tiempo_s"] * P["velocidad_ms"]

    n     = len(df)
    n_est = math.floor(n * P["pct_estacionario"])
    t_max = df["tiempo_s"].max()
    d_max = df["distancia_m"].max()

    # Frecuencia fija del ensayo (parámetro de entrada, por defecto 5 Hz)
    # La velocidad v es la velocidad media y solo se usa para d = v × t.
    v_mm_s             = P["velocidad_ms"] * 1000.0
    f_hz               = P["frecuencia_hz"]
    stroke_efectivo_mm = v_mm_s / (2.0 * f_hz)
    P["stroke_mm"]     = stroke_efectivo_mm
    n_ciclos           = round(f_hz * t_max)

    # Ventana en puntos: se calcula a partir de los segundos configurados
    # y la frecuencia de adquisición real → el suavizado es consistente
    # independientemente de cómo esté configurado el muestreo del tribómetro.
    hz_est = round(n / t_max) if t_max > 0 else 1
    w      = max(10, round(hz_est * P["ventana_s"]))
    # min_periods = w//2: evita el arranque inestable con 1–2 puntos en los
    # extremos; los valores iniciales usan al menos la mitad de la ventana.
    mp = max(1, w // 2)
    df["Fx_abs"]   = df["Fx_N"].abs()           # |Fx| instantáneo — ANTES de diezmar
    df["CoF_roll"] = df["CoF"].rolling(window=w, center=True, min_periods=mp).mean()
    df["Fx_roll"]  = df["Fx_abs"].rolling(window=w, center=True, min_periods=mp).mean()
    df["Z1_roll"]  = df["Z1_mm"].rolling(window=w, center=True, min_periods=mp).mean()

    print(f"  Suavizado:     {P['ventana_s']:.1f} s = {w} pts  (adq. {hz_est} Hz, min_periods={mp})")

    cof_med     = df["CoF"].mean()
    cof_sd      = df["CoF"].std()
    # mín/máx sobre la curva suavizada: el instantáneo está dominado por las
    # reversiones (Fx→0 ⇒ CoF→0) y picos de ruido, no es representativo.
    cof_min     = df["CoF_roll"].min()
    cof_max     = df["CoF_roll"].max()
    df_est      = df.iloc[n_est:]
    cof_est_med = df_est["CoF"].mean()
    cof_est_sd  = df_est["CoF"].std()
    dist_est    = df["distancia_m"].iloc[n_est]
    fz_med      = df["Fz_N"].mean()
    fz_sd       = df["Fz_N"].std()
    fx_med      = df["Fx_abs"].mean()
    fx_sd       = df["Fx_abs"].std()
    # hz_est ya calculado arriba para la ventana de suavizado

    # Z1: ventana fija de 100 puntos para init y final (evita deriva por n//100)
    n_avg       = min(100, max(10, n // 500))
    z1_init     = float(df["Z1_mm"].iloc[:n_avg].mean())
    z1_final    = float(df["Z1_mm"].iloc[-n_avg:].mean())
    z1_range    = z1_final - z1_init
    z1_rate     = abs(z1_range) / d_max if d_max > 0 else 0.0

    # Fin de rodadura: primer punto donde la curva suavizada alcanza el 95 %
    # del CoF estacionario. (El criterio anterior, CoF_roll > 0.05, se cumplía
    # ya en el primer punto y devolvía siempre 0 m.)
    mask_ri = df["CoF_roll"] >= 0.95 * cof_est_med
    d_runin = float(df.loc[mask_ri, "distancia_m"].iloc[0]) if mask_ri.any() else None

    step  = max(1, n // 5_000)
    # Diezmar por MEDIA de bin, no por muestreo puntual:
    # así df_p["CoF"] refleja el promedio real de cada bloque de 'step' puntos
    # y no depende de en qué fase del ciclo cae el punto muestreado.
    # CoF_roll (ya suavizado) no cambia visualmente, pero CoF instantáneo
    # queda alineado con lo que realmente pasó en ese intervalo.
    _bin = np.arange(n) // step
    df_p = df.groupby(_bin).mean().reset_index(drop=True)
    n3    = n // 3
    cof_t = [df["CoF"].iloc[:n3].mean(),
             df["CoF"].iloc[n3:2*n3].mean(),
             df["CoF"].iloc[2*n3:].mean()]

    print(f"  N registros:   {n:,}")
    print(f"  Duración:      {t_max/60:.1f} min  ({t_max:.0f} s)")
    print(f"  Distancia:     {d_max:.1f} m")
    print(f"  Frecuencia:    {f_hz:.2f} Hz  (fija del ensayo)  |  carrera efectiva: {stroke_efectivo_mm:.2f} mm")
    print(f"  Ciclos:        {n_ciclos:,}")
    print(f"  CoF estac.:    {cof_est_med:.4f} ± {cof_est_sd:.4f}")
    print(f"  Fz media:      {fz_med:.4f} ± {fz_sd:.4f} N")
    print(f"  ΔZ1:           {z1_range:+.4f} mm  (tasa {z1_rate*1e6:.1f} nm/m)")

    tabla = [
        ("ID ensayo",                          P["id_ensayo"]),
        ("Lote / muestra",                     P.get("lote", "")),
        ("Tratamiento",                        P.get("tratamiento", "")),
        ("Zona",                               P.get("zona", "")),
        ("Carga nominal",                      f"{P['carga_N']} N"),
        ("Velocidad media (v_media)",           f"{P['velocidad_ms']*1000:.0f} mm/s  ({P['velocidad_ms']} m/s)"),
        ("Frecuencia del ensayo (nominal)",    f"{f_hz:.2f} Hz"),
        ("Carrera efectiva (v / 2f)",          f"{stroke_efectivo_mm:.2f} mm"),
        ("N muestras",                         f"{n:,}"),
        ("Frecuencia adquisición",             f"{hz_est} Hz"),
        ("Ventana suavizado",                  f"{P['ventana_s']:.1f} s = {w} pts"),
        ("Duración real",                      f"{t_max:.0f} s  ({t_max/60:.1f} min)"),
        ("Distancia real",                     f"{d_max:.2f} m"),
        ("Número de ciclos",                   f"{n_ciclos:,.0f}"),
        ("Fz media",                           f"{fz_med:.4f} N"),
        ("Fz desv. estándar",                  f"{fz_sd:.4f} N"),
        ("CoF medio total",                    f"{cof_med:.4f}"),
        ("CoF desv. estándar total",           f"{cof_sd:.4f}"),
        ("CoF mínimo (suavizado)",             f"{cof_min:.4f}"),
        ("CoF máximo (suavizado)",             f"{cof_max:.4f}"),
        ("CoF medio — régimen estacionario",   f"{cof_est_med:.4f}"),
        ("CoF desv. est. — régimen estacion.", f"{cof_est_sd:.4f}"),
        ("Dist. inicio régimen estacionario",  f"{dist_est:.2f} m"),
        ("Dist. fin de rodadura",
         f"{d_runin:.2f} m" if d_runin is not None else "N/A"),
        ("Z1 inicial (media)",                 f"{z1_init:.4f} mm"),
        ("Z1 final (media)",                   f"{z1_final:.4f} mm"),
        ("ΔZ1 (desgaste vertical)",            f"{z1_range:+.4f} mm"),
        ("Tasa desgaste Z1",                   f"{z1_rate*1e6:.1f} nm/m"),
    ]

    stats = dict(
        cof_est_med=cof_est_med, cof_est_sd=cof_est_sd,
        cof_med=cof_med,         cof_sd=cof_sd,
        cof_min=cof_min,         cof_max=cof_max,
        fz_med=fz_med,           fz_sd=fz_sd,
        fx_med=fx_med,           fx_sd=fx_sd,
        t_max=t_max,             d_max=d_max,
        dist_est=dist_est,       d_runin=d_runin,
        cof_tercios=cof_t,       tabla=tabla,
        z1_init=z1_init,         z1_final=z1_final,
        z1_range=z1_range,       z1_rate=z1_rate,
        n_ciclos=n_ciclos,       f_hz=f_hz,
        hz_adq=hz_est,           n_muestras=n,
        anotaciones=P.get("anotaciones", ""),
    )

    return df, stats, df_p
