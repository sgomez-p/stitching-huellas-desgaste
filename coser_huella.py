#!/usr/bin/env python3
"""
coser_huella.py - Stitching de topografias opticas medidas a mano
===================================================================

Pensado para perfilometros confocales/interferometricos (p.ej. Sensofar PLu),
pero acepta cualquier topografia exportada como texto, GSF o .npy.

Junta varios campos medidos moviendo la platina XY manualmente (con solape),
los alinea en XY y en Z, nivela el mosaico respecto a la zona SIN desgastar
y calcula el volumen de desgaste (V-), el volumen apilado (V+) y el area de
la seccion transversal a lo largo de la huella.

USO BASICO
----------
    python coser_huella.py                       (abre ventana para elegir los campos)
    python coser_huella.py CARPETA_CON_CAMPOS
    python coser_huella.py c01.plu c02.plu ...

    Opciones utiles:
      --patron "*.plu"     ficheros a leer (por defecto: .plu .txt .asc .xyz .csv .dat .gsf .npy)
      --pixel 0.83         tamano de pixel en um (solo si el fichero no lo trae, p.ej. matriz ASCII)
      --ref-frac 0.15      fraccion de cada borde usada como superficie de referencia (sin desgaste)
      --ref-eje auto|x|y   bordes de referencia: 'y' = bandas arriba/abajo, 'x' = izquierda/derecha
      --barrido auto|x|y  direccion en que avanzan los campos (x: izq->dcha, y: arriba->abajo)
      --tilt               corrige tambien inclinacion relativa entre campos (por defecto solo offset Z)
      --sin-3d             no abrir el visor 3D al terminar (por defecto se abre en el navegador)
      --nombre NOMBRE      nombre de la huella (por defecto el de los ficheros sin el numero de parte:
                           MUESTRA-A-01-01, -01-02... -> MUESTRA-A-01)
      --salida DIR         carpeta de resultados (por defecto CARPETA/NOMBRE)
      --carga FN           carga normal (N) para la tasa de desgaste k (si no, se lee del nombre: '2N')
      --distancia D        distancia total deslizada (m) para k
      --carrera L          longitud de la huella / carrera (um) para k

Con un solo fichero no cose: nivela, calcula volumen y saca el 3D.

COMO MEDIR LOS CAMPOS
---------------------
  * Mismo objetivo, misma luz, mismo rango Z en todos los campos.
  * Solape de 25-30 % entre campos vecinos.
  * El orden y la direccion dan igual: el script busca solo donde encaja cada
    campo. Basta con que cada campo solape con alguno de los anteriores
    (orden de nombre de fichero: c01, c02, c03 ...).
  * Deja superficie sin desgastar a ambos lados de la huella (es la referencia).

FORMATOS
--------
  .plu   Sensofar PLu (lectura heuristica del formato antiguo; si falla,
         exporta desde SensoSCAN/SensoMAP como texto)
  .txt/.asc/.csv/.dat  matriz de alturas (um)  -> requiere --pixel
                       o tres columnas X Y Z (um) -> pixel automatico
  .xyz   tres columnas X Y Z (um)
  .gsf   Gwyddion Simple Field
  .npy   matriz numpy (um) -> requiere --pixel

SALIDAS (en CARPETA/NOMBRE/, cada fichero empieza por NOMBRE_ , p.ej. MUESTRA-A-01_mosaico.png)
------------------------------------------------------------------------------------------------
  mosaico.png            mapa de alturas nivelado + contorno de cada campo
  mosaico_3d.png         vista 3D
  mosaico_3d.html        visor 3D interactivo (plotly): girar/zoom con el raton, botones de
                         exageracion Z, vistas y colores; el icono de camara guarda PNG en alta
                         resolucion con el encuadre actual. Funciona sin internet.
  perfil_central.png     perfil transversal medio a lo largo de la huella (+ .csv para Origin)
  area_seccion.csv/.png  area de la seccion de desgaste (y k) a lo largo de la huella
  resumen.csv            una fila: areas medias, tasa de desgaste especifica k, prof. maxima
  mosaico.gsf            mosaico para abrir en Gwyddion (y de ahi a MountainsMap)
  mosaico_matriz.txt     matriz ASCII de alturas (um)
  informe.txt            posiciones, calidad del cosido, offsets y volumenes

Dependencias: numpy, scipy, matplotlib  (scikit-image y plotly opcionales)
    pip install -r requirements.txt

Funciona en Windows, macOS y Linux (x86-64 y ARM/Apple Silicon).
"""

import argparse
import glob
import os
import re
import struct
import subprocess
import sys

import numpy as np
from scipy import ndimage, fft as sfft

# Consolas de Windows (cp1252): que un print con µ, ±, → nunca aborte el script
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def _cross_correlate_masked_np(arr1, arr2, m1, m2, overlap_ratio=0.3):
    """Correlacion cruzada normalizada enmascarada (Padfield, IEEE TIP 2012),
    modo 'full'. Implementacion propia con scipy.fft, equivalente a la de
    scikit-image, para no depender de un modulo privado que cambia entre versiones."""
    eps = np.finfo(float).eps
    m1 = np.asarray(m1, bool); m2 = np.asarray(m2, bool)
    a1 = np.where(m1, arr1, 0.0); a2 = np.where(m2, arr2, 0.0)
    rot = lambda x: x[::-1, ::-1]
    a2r, m2r = rot(a2), rot(m2.astype(float))
    shape = tuple(a + b - 1 for a, b in zip(a1.shape, a2.shape))
    fshape = tuple(sfft.next_fast_len(n, real=True) for n in shape)
    F = lambda x: sfft.rfftn(x, fshape)
    iF = lambda X: sfft.irfftn(X, fshape)[tuple(slice(0, n) for n in shape)]
    f1, fm1 = F(a1), F(m1.astype(float))
    f2, fm2 = F(a2r), F(m2r)
    n_ov = np.maximum(np.round(iF(fm1 * fm2)), eps)
    num = iF(f1 * f2) - iF(f1 * fm2) * iF(fm1 * f2) / n_ov
    d1 = iF(F(a1 * a1) * fm2) - iF(f1 * fm2) ** 2 / n_ov
    d2 = iF(fm1 * F(a2r * a2r)) - iF(fm1 * f2) ** 2 / n_ov
    den = np.sqrt(np.maximum(d1 * d2, 0.0))
    out = np.zeros_like(num)
    tol = 1e3 * eps * np.max(np.abs(den))
    ok = den > tol
    out[ok] = num[ok] / den[ok]
    np.clip(out, -1, 1, out=out)
    out[n_ov < overlap_ratio * np.max(n_ov)] = 0.0
    return out


try:   # si esta scikit-image se usa su version (mas probada); si no, la propia
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
except Exception:
    def cross_correlate_masked(a1, a2, m1, m2, mode="full", axes=(-2, -1), overlap_ratio=0.3):
        return _cross_correlate_masked_np(a1, a2, m1, m2, overlap_ratio)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------
# LECTURA DE FICHEROS
# ----------------------------------------------------------------------------

def leer_plu(path):
    """Lector heuristico del formato Sensofar PLu antiguo (PLu 2300).

    Estructura (segun la documentacion del formato usada por Gwyddion):
      128 B fecha | 4 B time_t | 256 B comentario | bloque calibracion:
      xres, yres, N_tall (uint32), dy_multip, mppx, mppy, x0, y0, mpp_tall, z0 (float32)
      ... bloque de configuracion ... | yres, xres (uint32) | datos float32
    Los puntos no medidos llevan un valor enorme (~1e6) -> NaN.
    En lugar de fiarnos de un offset fijo, buscamos el par (filas, columnas)
    que deja exactamente el numero de bytes de datos esperado.
    """
    raw = open(path, "rb").read()
    n = len(raw)
    off_cal = 128 + 4 + 256
    mppx = mppy = None
    try:
        xres_c, yres_c, _ = struct.unpack_from("<3I", raw, off_cal)
        _, mppx, mppy = struct.unpack_from("<3f", raw, off_cal + 12)
        if not (1e-4 < mppx < 100 and 1e-4 < mppy < 100):
            mppx = mppy = None
    except struct.error:
        pass

    # 1) Formato verificado con ficheros reales del PLu 2300 (SensoSCAN):
    #    filas, columnas en off_cal; la pareja (filas, columnas) se repite justo
    #    antes de los datos (offset 500 -> datos en 508). Tras las alturas hay
    #    mas bloques (imagen de intensidad...) que no usamos.
    try:
        filas, cols = struct.unpack_from("<2I", raw, off_cal)
        ultimo = None
        for o in range(off_cal + 40, min(n - 8, 2048), 4):
            if struct.unpack_from("<2I", raw, o) == (filas, cols) and n - (o + 8) >= filas * cols * 4:
                ultimo = o
        if ultimo is not None and 16 <= filas <= 20000 and 16 <= cols <= 20000:
            z = np.frombuffer(raw, dtype="<f4", count=filas * cols, offset=ultimo + 8)
            z = z.astype(float).reshape(filas, cols)
            z = np.where((np.abs(z) > 1e5) | ~np.isfinite(z), np.nan, z)
            return z, (mppx if mppx else None)
    except struct.error:
        pass

    # 2) Heuristica de respaldo para otras versiones del formato
    candidatos = []
    for o in range(off_cal, min(n - 8, 4096), 4):
        a, b = struct.unpack_from("<2I", raw, o)
        if not (16 <= a <= 20000 and 16 <= b <= 20000):
            continue
        datos = a * b * 4
        resto = n - (o + 8) - datos
        # resto 0, o una imagen de intensidad de 1 B/pixel, o una cola pequena
        if resto in (0, a * b) or 0 <= resto <= 64:
            candidatos.append((o, a, b, resto))
    if not candidatos:
        raise ValueError(f"No reconozco la estructura de {os.path.basename(path)}. "
                         "Exportalo como texto (matriz o XYZ) desde SensoSCAN/SensoMAP.")
    o, a, b, _ = candidatos[0]
    z = np.frombuffer(raw, dtype="<f4", count=a * b, offset=o + 8).astype(float)
    # Gwyddion lee yres, xres en ese orden; probamos ambas orientaciones y
    # nos quedamos con la mas "suave" entre filas (la correcta).
    z1 = z.reshape(a, b)
    z2 = z.reshape(b, a)
    def rugosidad_filas(m):
        m = np.where(np.abs(m) > 1e5, np.nan, m)
        return np.nanmedian(np.abs(np.diff(m, axis=0)))
    z = z1 if rugosidad_filas(z1) <= rugosidad_filas(z2) else z2
    z = np.where((np.abs(z) > 1e5) | ~np.isfinite(z), np.nan, z)
    px = mppx if mppx else None
    if mppx and mppy and abs(mppx - mppy) / mppx > 0.02:
        print(f"  aviso: pixel no cuadrado en {os.path.basename(path)} ({mppx:.4f} x {mppy:.4f} um); uso X")
    return z, px


def leer_gsf(path):
    raw = open(path, "rb").read()
    fin = raw.index(b"\x00")
    cab = raw[:fin].decode("utf-8", "replace").splitlines()
    meta = {}
    for linea in cab[1:]:
        if "=" in linea:
            k, v = linea.split("=", 1)
            meta[k.strip()] = v.strip()
    xres, yres = int(meta["XRes"]), int(meta["YRes"])
    inicio = fin + (4 - fin % 4)
    z = np.frombuffer(raw, dtype="<f4", count=xres * yres, offset=inicio).reshape(yres, xres)
    escala_z = 1e6 if meta.get("ZUnits", "m") == "m" else 1.0
    escala_xy = 1e6 if meta.get("XYUnits", "m") == "m" else 1.0
    px = float(meta["XReal"]) * escala_xy / xres
    return z.astype(float) * escala_z, px


def leer_ascii(path):
    # admite separadores coma, punto y coma, tab o espacio; ignora cabeceras
    filas = []
    with open(path, "r", errors="replace") as f:
        for linea in f:
            linea = linea.strip().replace(";", " ").replace(",", " ").replace("\t", " ")
            if not linea:
                continue
            try:
                filas.append([float(v) if v.lower() not in ("nan", "***", "-") else np.nan
                              for v in linea.split()])
            except ValueError:
                continue  # cabecera
    if not filas:
        raise ValueError(f"{path}: no hay datos numericos")
    ncol = max(len(r) for r in filas)
    filas = [r for r in filas if len(r) == ncol]
    m = np.array(filas, dtype=float)
    if ncol == 3 and m.shape[0] > 100:
        # XYZ -> rejilla
        xs = np.unique(np.round(m[:, 0], 6))
        ys = np.unique(np.round(m[:, 1], 6))
        px = float(np.median(np.diff(xs)))
        ix = np.round((m[:, 0] - xs.min()) / px).astype(int)
        iy = np.round((m[:, 1] - ys.min()) / px).astype(int)
        z = np.full((iy.max() + 1, ix.max() + 1), np.nan)
        z[iy, ix] = m[:, 2]
        return z, px
    return m, None


def nombre_huella(ficheros):
    """Nombre de la huella a partir de los ficheros de sus partes.

    El ultimo numero de cada fichero es la parte: MUESTRA-A-01-01, -01-02 ...
    -> MUESTRA-A-01. Con un solo fichero se usa su nombre tal cual.
    """
    nombres = [os.path.splitext(os.path.basename(f))[0] for f in ficheros]
    if len(nombres) == 1:
        return nombres[0]
    bases = {re.sub(r"[\s_-]*\d+$", "", n).rstrip(" -_") for n in nombres}
    if len(bases) == 1 and bases != {""}:
        return bases.pop()
    comun = os.path.commonprefix(nombres).rstrip(" -_")
    return comun or nombres[0]


def carga_de_ruta(ruta):
    """Carga normal (N) leida del nombre del fichero o de sus carpetas:
    'ENSAYO-1N-01' -> 1, 'CARGA 0.5N' -> 0.5, 'M1 0_5N - 01' -> 0.5."""
    partes = [os.path.splitext(os.path.basename(ruta))[0]] + \
             list(reversed(os.path.dirname(os.path.abspath(ruta)).split(os.sep)))
    for p in partes:
        m = re.search(r"(?<![\d.,_])(\d+(?:[.,_]\d+)?)\s*N(?![a-zA-Z])", p)
        if m:
            return float(m.group(1).replace(",", ".").replace("_", "."))
    return None


def k_archard(area_um2, carrera_um, fn_n, d_m):
    """Tasa de desgaste especifica k = V / (Fn*d), con V = Area_seccion * carrera.
    En x10^-6 mm3/(N*m)."""
    return area_um2 * carrera_um * 1e-3 / (fn_n * d_m)


def leer_campo(path, pixel_cli):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".plu":
        z, px = leer_plu(path)
    elif ext == ".gsf":
        z, px = leer_gsf(path)
    elif ext == ".npy":
        z, px = np.load(path).astype(float), None
    else:
        z, px = leer_ascii(path)
    if pixel_cli:
        px = pixel_cli
    if px is None:
        sys.exit(f"ERROR: {os.path.basename(path)} no trae tamano de pixel. Usa --pixel (um).")
    return z, px


# ----------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------

def orden_natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def rellenar_nan(z, sigma=5):
    """Rellena NaN con la media local (convolucion normalizada)."""
    v = np.isfinite(z)
    z0 = np.where(v, z, 0.0)
    num = ndimage.gaussian_filter(z0, sigma)
    den = ndimage.gaussian_filter(v.astype(float), sigma)
    rel = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nanmean(z))
    return np.where(v, z, rel)


def paso_alto(z, sigma=25):
    """Quita la forma/inclinacion y deja la textura, que es lo que se correlaciona."""
    zf = rellenar_nan(z)
    hp = zf - ndimage.gaussian_filter(zf, sigma)
    s = np.nanstd(hp)
    return hp / s if s > 0 else hp


def quitar_picos(z, k=8.0, tam=5, minimo=0.5):
    """Pone a NaN los picos aislados (outliers tipicos de confocal/VSI):
    puntos que se separan de la mediana local mas de k sigmas robustas."""
    zf = rellenar_nan(z, 3)
    med = ndimage.median_filter(zf, size=tam)
    d = z - med
    sig = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))
    picos = np.abs(d) > max(k * sig, minimo)
    return np.where(picos, np.nan, z), int(np.nansum(picos))


def peso_borde(shape):
    """Peso de fundido (feathering): 1 en el centro, decae hacia los bordes."""
    h, w = shape
    y = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    return np.minimum.outer(y, x).astype(float)


def solape(pos_a, shp_a, pos_b, shp_b):
    """Rectangulo comun (coordenadas globales) entre dos campos."""
    y0 = max(pos_a[0], pos_b[0]); x0 = max(pos_a[1], pos_b[1])
    y1 = min(pos_a[0] + shp_a[0], pos_b[0] + shp_b[0])
    x1 = min(pos_a[1] + shp_a[1], pos_b[1] + shp_b[1])
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, x0, y1, x1


def ncc_solape(hp_a, pos_a, hp_b, pos_b):
    r = solape(pos_a, hp_a.shape, pos_b, hp_b.shape)
    if r is None:
        return -1.0, 0
    y0, x0, y1, x1 = r
    a = hp_a[y0 - pos_a[0]:y1 - pos_a[0], x0 - pos_a[1]:x1 - pos_a[1]]
    b = hp_b[y0 - pos_b[0]:y1 - pos_b[0], x0 - pos_b[1]:x1 - pos_b[1]]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / d) if d > 0 else -1.0), a.size


# ----------------------------------------------------------------------------
# COSIDO
# ----------------------------------------------------------------------------

def desplazamiento(hp_j, m_j, hp_i, m_i, barrido, solape_min, deriva=0.2):
    """Mejor posicion del campo i respecto al j (dy, dx en px) y su correlacion.

    Calcula el mapa completo de correlacion enmascarada (tolera NaN) y solo
    acepta desplazamientos compatibles con el barrido:
      'x' -> los campos avanzan en X (izquierda/derecha); en Y solo una pequena deriva
      'y' -> avanzan en Y (arriba/abajo); en X solo una pequena deriva
    Asi no confunde una huella con la de al lado por parecerse las rayas.
    """
    m = cross_correlate_masked(hp_j, hp_i, m_j, m_i, mode="full", axes=(-2, -1),
                               overlap_ratio=solape_min)
    H, W = hp_i.shape
    dy = (np.arange(m.shape[0]) - H + 1)[:, None]
    dx = (np.arange(m.shape[1]) - W + 1)[None, :]
    frac = (np.clip(np.minimum(hp_j.shape[0], dy + H) - np.maximum(0, dy), 0, None) *
            np.clip(np.minimum(hp_j.shape[1], dx + W) - np.maximum(0, dx), 0, None)) / (H * W)
    ok = (frac >= solape_min) & np.isfinite(m)
    if barrido == "x":
        ok &= np.abs(dy) <= deriva * H
    elif barrido == "y":
        ok &= np.abs(dx) <= deriva * W
    if not ok.any():
        return None
    k = np.unravel_index(np.argmax(np.where(ok, m, -np.inf)), m.shape)
    return np.array([dy[k[0], 0], dx[0, k[1]]]), float(frac[k])


def registrar(campos, barrido="auto", solape_min=0.05, ncc_min=0.4, forzar=False, preguntar=None):
    """Coloca cada campo respecto al mejor vecino ya colocado.

    La posicion se busca con la textura a escala media (paso alto sigma 8) y se
    valida con la textura fina (sigma 3): la forma de la huella se parece entre
    huellas distintas, la textura fina solo coincide si los campos solapan de verdad.
    """
    hps = [paso_alto(z, 8) for z, _ in campos]
    finos = [paso_alto(z, 3) for z, _ in campos]
    masks = [np.isfinite(z) for z, _ in campos]
    pos = [np.array([0, 0])]
    calidad = [(None, 1.0)]
    for i in range(1, len(campos)):
        mejor = None
        for j in range(i - 1, -1, -1):   # primero los mas recientes
            r = desplazamiento(hps[j], masks[j], hps[i], masks[i], barrido, solape_min)
            if r is None:
                continue
            d, frac = r
            p = pos[j] + d
            ncc, _ = ncc_solape(finos[j], pos[j], finos[i], p)
            if mejor is None or ncc > mejor[0]:
                mejor = (ncc, j, p, frac)
            if ncc > 0.8:   # encaje claro: no hace falta probar mas
                break
        if mejor is not None and mejor[0] < ncc_min and not forzar and preguntar is not None:
            # modo ventana: en vez de parar, se pregunta si coser igualmente
            forzar = preguntar(f"El campo {i+1} no encaja bien con ninguno anterior "
                               f"(correlacion de textura {mejor[0]:.2f}; con solape real sale > 0.6).\n\n"
                               "Puede que no solapen o que sean huellas distintas.\n\n"
                               "¿Coser igualmente? (el resultado puede ser incorrecto)")
            if not forzar:
                sys.exit("Cancelado: no se cosio porque los campos no encajan.")
        if mejor is None or (mejor[0] < ncc_min and not forzar):
            extra = f" (mejor correlacion de textura {mejor[0]:.2f}; con solape real sale > 0.6)" if mejor else ""
            sys.exit(f"ERROR: el campo {i+1} no encaja con ninguno anterior{extra}.\n"
                     "  No comparten zona medida: puede que sean huellas distintas o que falte solape.\n"
                     "  Al barrer la huella, deja un 25-30 % de solape entre campos consecutivos.\n"
                     "  Si quieres coserlos igualmente, usa --forzar.")
        ncc, j, p, frac = mejor
        pos.append(p)
        calidad.append((j, ncc))
        aviso = "  <-- REVISAR (correlacion baja)" if ncc < 0.5 else ""
        print(f"  campo {i+1:2d} -> sobre campo {j+1:2d}  dx={p[1]-pos[j][1]:+5d} px "
              f"dy={p[0]-pos[j][0]:+5d} px  solape={frac*100:4.1f}%  NCC={ncc:.3f}{aviso}")
    return pos, calidad


def ajustar_z(campos, pos, con_tilt=False):
    """Calcula el offset en Z (y opcionalmente la inclinacion) de cada campo
    respecto a los ya colocados, usando la mediana de la diferencia en el solape."""
    zs = [z.copy() for z, _ in campos]
    correcciones = [(0.0, 0.0, 0.0)]
    for i in range(1, len(zs)):
        dif, yy, xx = [], [], []
        for j in range(i):
            r = solape(pos[j], zs[j].shape, pos[i], zs[i].shape)
            if r is None:
                continue
            y0, x0, y1, x1 = r
            a = zs[j][y0 - pos[j][0]:y1 - pos[j][0], x0 - pos[j][1]:x1 - pos[j][1]]
            b = zs[i][y0 - pos[i][0]:y1 - pos[i][0], x0 - pos[i][1]:x1 - pos[i][1]]
            v = np.isfinite(a) & np.isfinite(b)
            if v.sum() < 50:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1]
            dif.append((a - b)[v]); yy.append(gy[v]); xx.append(gx[v])
        if not dif:
            correcciones.append((0.0, 0.0, 0.0))
            continue
        d = np.concatenate(dif); gy = np.concatenate(yy); gx = np.concatenate(xx)
        if con_tilt:
            # plano robusto: un ajuste, quita atipicos, reajuste
            A = np.c_[np.ones_like(d), gx - pos[i][1], gy - pos[i][0]]
            c, *_ = np.linalg.lstsq(A, d, rcond=None)
            with np.errstate(all="ignore"):   # avisos espurios de numpy 2 + Accelerate (macOS)
                res = d - A @ c
            ok = np.abs(res) < 3 * np.std(res)
            c, *_ = np.linalg.lstsq(A[ok], d[ok], rcond=None)
        else:
            c = np.array([np.median(d), 0.0, 0.0])
        h, w = zs[i].shape
        Y, X = np.mgrid[0:h, 0:w]
        zs[i] = zs[i] + c[0] + c[1] * X + c[2] * Y
        correcciones.append(tuple(c))
    return zs, correcciones


def componer(zs, pos):
    p = np.array(pos)
    p = p - p.min(axis=0)
    H = max(p[k][0] + zs[k].shape[0] for k in range(len(zs)))
    W = max(p[k][1] + zs[k].shape[1] for k in range(len(zs)))
    suma = np.zeros((H, W)); pesos = np.zeros((H, W))
    for z, (y, x) in zip(zs, p):
        w = peso_borde(z.shape) * np.isfinite(z)
        suma[y:y + z.shape[0], x:x + z.shape[1]] += np.nan_to_num(z) * w
        pesos[y:y + z.shape[0], x:x + z.shape[1]] += w
    mos = np.where(pesos > 0, suma / np.maximum(pesos, 1e-12), np.nan)
    return mos, p


# ----------------------------------------------------------------------------
# NIVELADO Y VOLUMEN
# ----------------------------------------------------------------------------

def orientacion_huella(mos):
    """Detecta hacia donde va la huella: quita un plano y compara cuanto varia
    el perfil medio de columnas frente al de filas. Si las columnas varian mas,
    la huella va en Y (vertical) -> referencia a izquierda/derecha ('x')."""
    H, W = mos.shape
    Y, X = np.mgrid[0:H, 0:W]
    v = np.isfinite(mos)
    A = np.c_[np.ones(v.sum()), X[v], Y[v]]
    c, *_ = np.linalg.lstsq(A, mos[v], rcond=None)
    r = mos - (c[0] + c[1] * X + c[2] * Y)
    var_cols = np.nanstd(np.nanmean(r, axis=0))
    var_filas = np.nanstd(np.nanmean(r, axis=1))
    return "x" if var_cols > var_filas else "y"


def mascara_referencia(shape, frac, eje, mos=None):
    H, W = shape
    if eje == "auto":
        eje = orientacion_huella(mos) if mos is not None else ("y" if W >= H else "x")
    m = np.zeros(shape, bool)
    if eje == "y":
        n = max(1, int(H * frac)); m[:n] = True; m[-n:] = True
    else:
        n = max(1, int(W * frac)); m[:, :n] = True; m[:, -n:] = True
    return m, eje


def nivelar(mos, ref):
    H, W = mos.shape
    Y, X = np.mgrid[0:H, 0:W]
    v = ref & np.isfinite(mos)
    A = np.c_[np.ones(v.sum()), X[v], Y[v]]
    d = mos[v]
    ok = np.ones(d.size, bool)
    for _ in range(3):   # plano robusto (excluye picos/defectos en la referencia)
        c, *_ = np.linalg.lstsq(A[ok], d[ok], rcond=None)
        with np.errstate(all="ignore"):   # avisos espurios de numpy 2 + Accelerate (macOS)
            res = d - A @ c
        ok = np.abs(res - np.median(res)) < 3 * np.std(res[ok])
    niv = mos - (c[0] + c[1] * X + c[2] * Y)
    sigma_ref = float(np.std(res[ok]))
    return niv, sigma_ref


def volumenes(niv, ref, px, sigma_ref, k_umbral=3.0):
    zona = ~ref & np.isfinite(niv)
    z = niv[zona]
    a = px * px
    v_menos = float(np.sum(-z[z < 0]) * a)
    v_mas = float(np.sum(z[z > 0]) * a)
    u = k_umbral * sigma_ref
    v_menos_u = float(np.sum(-z[z < -u]) * a)
    v_mas_u = float(np.sum(z[z > u]) * a)
    return dict(V_menos=v_menos, V_mas=v_mas, V_menos_umbral=v_menos_u,
                V_mas_umbral=v_mas_u, umbral=u, prof_max=float(-np.nanmin(z)))


def area_seccion(niv, ref, px, eje, sigma_ref, k_umbral=3.0):
    """Area de la seccion de desgaste (um^2) a lo largo de la huella."""
    z = np.where(~ref & np.isfinite(niv), niv, np.nan)
    u = k_umbral * sigma_ref
    perdida = np.where(z < -u, -z, 0.0)
    # eje 'y' = bandas arriba/abajo -> la huella va en X -> secciones = columnas
    if eje == "y":
        A = np.nansum(perdida, axis=0) * px
        valid = np.isfinite(z).sum(axis=0) > 0.5 * z.shape[0] * 0.5
    else:
        A = np.nansum(perdida, axis=1) * px
        valid = np.isfinite(z).sum(axis=1) > 0.5 * z.shape[1] * 0.5
    s = np.arange(A.size) * px
    return s, A, valid


# ----------------------------------------------------------------------------
# SALIDAS
# ----------------------------------------------------------------------------

def guardar_gsf(path, z, px):
    H, W = z.shape
    cab = (f"Gwyddion Simple Field 1.0\nXRes = {W}\nYRes = {H}\n"
           f"XReal = {W*px*1e-6:.9g}\nYReal = {H*px*1e-6:.9g}\n"
           f"XYUnits = m\nZUnits = m\nTitle = mosaico\n").encode()
    pad = 4 - len(cab) % 4
    with open(path, "wb") as f:
        f.write(cab + b"\x00" * pad)
        f.write((z * 1e-6).astype("<f4").tobytes())


def visor_3d(niv, px, lim, path, max_lado=900):
    """Visor 3D interactivo (HTML autonomo, funciona sin internet).

    Raton: girar (arrastrar), zoom (rueda), desplazar (clic derecho / shift).
    Botones: exageracion Z, vistas predefinidas, mapa de color.
    Icono de camara de la barra superior: guarda PNG a 4x la resolucion de pantalla
    con el encuadre actual (para capturas de articulo/presentacion).
    """
    import plotly.graph_objects as go
    H, W = niv.shape
    paso = max(1, int(np.ceil(max(H, W) / max_lado)))
    zz = np.clip(rellenar_nan(niv), *lim)[::paso, ::paso]
    x = np.arange(zz.shape[1]) * paso * px / 1000
    y = np.arange(zz.shape[0]) * paso * px / 1000
    lx, ly = x.max(), y.max()
    # proporcion real: rango Z (um -> mm) frente al lado mayor (mm)
    z_real = (lim[1] - lim[0]) / 1000 / max(lx, ly)
    base = dict(x=lx / max(lx, ly), y=ly / max(lx, ly))

    fig = go.Figure(go.Surface(
        x=x, y=y, z=zz, colorscale="Viridis", cmin=lim[0], cmax=lim[1],
        colorbar=dict(title="z (µm)", len=0.7),
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
        hovertemplate="x=%{x:.3f} mm<br>y=%{y:.3f} mm<br>z=%{z:.3f} µm<extra></extra>"))

    def aspecto(f):
        return dict(base, z=f)
    exag = [("Z real", z_real)] + [(f"Z x{k}", z_real * k) for k in (5, 10, 20, 50)] + [("Z auto", 0.35)]
    exag = [(n, f) for n, f in exag if f <= 3 or n == "Z real"]
    vistas = [("Isometrica", dict(eye=dict(x=-1.4, y=-1.6, z=1.1))),
              ("Arriba", dict(eye=dict(x=0, y=0, z=2.4), up=dict(x=0, y=1, z=0))),
              ("Lateral X", dict(eye=dict(x=0, y=-2.4, z=0.15))),
              ("Lateral Y", dict(eye=dict(x=2.4, y=0, z=0.15)))]
    colores = ["Viridis", "Turbo", "Jet", "Hot", "RdBu", "Cividis", "Greys"]

    fig.update_layout(
        title="Huella cosida - arrastra para girar, rueda para zoom",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (µm)",
                   aspectmode="manual", aspectratio=aspecto(0.35),
                   camera=vistas[0][1]),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", yanchor="top",
                 buttons=[dict(label=n, method="relayout", args=[{"scene.aspectratio": aspecto(f)}])
                          for n, f in exag]),
            dict(type="buttons", direction="right", x=0.0, y=0.94, xanchor="left", yanchor="top",
                 buttons=[dict(label=n, method="relayout", args=[{"scene.camera": c}])
                          for n, c in vistas]),
            dict(type="buttons", direction="right", x=0.0, y=0.88, xanchor="left", yanchor="top",
                 buttons=[dict(label=c, method="restyle", args=[{"colorscale": c}])
                          for c in colores]),
        ])
    fig.write_html(path, include_plotlyjs=True, full_html=True, config=dict(
        displaylogo=False, scrollZoom=True,
        toImageButtonOptions=dict(format="png", filename="huella_3d", scale=4)))


def figuras(niv, pos_px, shapes, px, eje, ref, s, A, valid, ruta):
    H, W = niv.shape
    ext = [0, W * px / 1000, H * px / 1000, 0]  # mm
    lim = np.nanpercentile(niv, [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(10, 10 * H / W + 1.2))
    im = ax.imshow(niv, cmap="viridis", extent=ext, vmin=lim[0], vmax=lim[1])
    for k, ((y, x), (h, w)) in enumerate(zip(pos_px, shapes)):
        ax.add_patch(plt.Rectangle((x * px / 1000, y * px / 1000), w * px / 1000, h * px / 1000,
                                   fill=False, ec="w", lw=0.6, alpha=0.6))
        ax.text((x + w / 2) * px / 1000, (y + h / 2) * px / 1000, str(k + 1),
                color="w", ha="center", va="center", fontsize=8, alpha=0.8)
    ax.contour(np.linspace(0, ext[1], W), np.linspace(0, ext[2], H), ref.astype(float),
               levels=[0.5], colors="r", linewidths=0.8, linestyles="--")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Mosaico nivelado (rojo discontinuo: limite de la zona de referencia)")
    plt.colorbar(im, ax=ax, label="z (µm)", shrink=0.8)
    fig.tight_layout(); fig.savefig(ruta("mosaico.png"), dpi=200); plt.close(fig)

    # 3D (submuestreado)
    paso = max(1, int(max(H, W) / 400))
    zz = np.clip(rellenar_nan(niv), *lim)[::paso, ::paso]
    Y, X = np.mgrid[0:zz.shape[0], 0:zz.shape[1]] * paso * px / 1000
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, zz, cmap="viridis", vmin=lim[0], vmax=lim[1],
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_zlim(lim[0] - 0.1 * (lim[1] - lim[0]), lim[1] + 0.1 * (lim[1] - lim[0]))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (µm)")
    ax.set_box_aspect((X.max(), Y.max(), 0.35 * max(X.max(), Y.max())))
    ax.view_init(elev=35, azim=-60)
    ax.set_title("Vista 3D (escala Z exagerada)")
    fig.tight_layout(); fig.savefig(ruta("mosaico_3d.png"), dpi=200); plt.close(fig)

    try:
        visor_3d(niv, px, lim, ruta("mosaico_3d.html"))
    except ImportError:
        print("  (instala plotly para tener tambien el 3D interactivo: pip install plotly)")
    except Exception as e:
        print(f"  aviso: no se pudo generar el 3D interactivo ({e})")

    # perfil transversal: uno individual (gris) + el medio a lo largo de la huella (negro)
    Av = np.where(valid, A, 0)
    k = int(np.argmax(Av))
    mat = niv if eje == "y" else niv.T          # columnas = secciones a lo largo de la huella
    sel = valid & (A > 0.1 * Av.max()) if Av.max() > 0 else valid
    medio = np.nanmean(mat[:, sel], axis=1) if sel.any() else mat[:, k]
    desv = np.nanstd(mat[:, sel], axis=1) if sel.any() else np.zeros_like(medio)
    coord = np.arange(mat.shape[0]) * px / 1000
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(coord, mat[:, k], lw=0.5, color="0.65", label="perfil individual")
    ax.fill_between(coord, medio - desv, medio + desv, color="C0", alpha=0.25, lw=0, label="± desv. estandar")
    ax.plot(coord, medio, lw=1.2, color="k", label=f"perfil medio ({int(sel.sum())} secciones)")
    ax.axhline(0, color="r", lw=0.6, ls="--")
    ax.set_xlabel("posicion transversal (mm)"); ax.set_ylabel("z (µm)")
    ax.set_title("Perfil transversal de la huella")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(ruta("perfil_central.png"), dpi=200); plt.close(fig)
    np.savetxt(ruta("perfil_medio.csv"), np.c_[coord * 1000, medio, desv],
               delimiter=",", fmt="%.4f", header="posicion_um,z_medio_um,desv_um", comments="")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(s[valid] / 1000, A[valid], lw=0.8, color="k")
    ax.set_xlabel("posicion a lo largo de la huella (mm)"); ax.set_ylabel("area de seccion (µm²)")
    ax.set_title("Area de la seccion de desgaste")
    fig.tight_layout(); fig.savefig(ruta("area_seccion.png"), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------
# PRINCIPAL
# ----------------------------------------------------------------------------

EXTENSIONES = (".plu", ".txt", ".asc", ".xyz", ".csv", ".dat", ".gsf", ".npy")


# Dialogos: en macOS se usan los nativos del sistema (AppleScript), porque el Tk
# de conda en Mac dibuja mal las ventanas y no deja arrastrarlas. En otros
# sistemas, tkinter.
MAC = sys.platform == "darwin"
MODO_GUI = False
_RAIZ = None


def _osa(script, *args):
    """Ejecuta AppleScript (con 'on run argv'). Devuelve el texto o None si se cancela."""
    r = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True)
    if r.returncode != 0:
        if "-128" in r.stderr:   # el usuario pulso Cancelar
            return None
        raise RuntimeError(f"osascript: {r.stderr.strip()}")
    return r.stdout.rstrip("\n")


def _tk_raiz():
    global _RAIZ
    if _RAIZ is None:
        try:
            import tkinter as tk
            _RAIZ = tk.Tk()
        except Exception as e:   # Linux sin tkinter o sin pantalla
            sys.exit("ERROR: no se puede abrir la ventana de seleccion (tkinter/pantalla no disponible: "
                     f"{e}).\n  Pasa los ficheros por linea de comandos: python coser_huella.py CARPETA\n"
                     "  (en Debian/Ubuntu: sudo apt install python3-tk)")
        _RAIZ.withdraw()
    return _RAIZ


def cerrar_gui():
    global _RAIZ
    if _RAIZ is not None:
        _RAIZ.destroy()
        _RAIZ = None


OSA_FICHEROS = """on run argv
    activate
    set fs to choose file with prompt (item 1 of argv) with multiple selections allowed
    set out to ""
    repeat with f in fs
        set out to out & POSIX path of f & linefeed
    end repeat
    return out
end run"""

OSA_PIXEL = """on run argv
    activate
    return text returned of (display dialog (item 1 of argv) default answer "" with title "Tamano de pixel")
end run"""

OSA_BARRIDO = """on run argv
    activate
    return button returned of (display dialog (item 1 of argv) with title "Direccion del barrido" buttons {item 2 of argv, item 3 of argv, item 4 of argv} default button 2)
end run"""

OSA_AVISO = """on run argv
    activate
    if item 3 of argv is "error" then
        display alert (item 1 of argv) message (item 2 of argv) as critical
    else
        display alert (item 1 of argv) message (item 2 of argv) as informational
    end if
end run"""


def elegir_ficheros():
    """Ventana para seleccionar los campos a coser (seleccion multiple)."""
    if MAC:
        r = _osa(OSA_FICHEROS, "Selecciona los campos a coser (Cmd o Shift para elegir varios)")
        if not r:
            return []
        ficheros = [f for f in r.split("\n") if f]
        ignorados = [f for f in ficheros if os.path.splitext(f)[1].lower() not in EXTENSIONES]
        if ignorados:
            print("  ignorados (formato no soportado): " + ", ".join(os.path.basename(f) for f in ignorados))
        return [f for f in ficheros if f not in ignorados]
    from tkinter import filedialog
    tipos = [("Topografias", " ".join("*" + e for e in EXTENSIONES)), ("Todos", "*")]
    return list(filedialog.askopenfilenames(parent=_tk_raiz(), title="Selecciona los campos a coser",
                                            filetypes=tipos))


OSA_NOMBRE = """on run argv
    activate
    return text returned of (display dialog (item 1 of argv) default answer (item 2 of argv) with title "Nombre de la huella")
end run"""


def pedir_nombre(propuesto):
    texto = ("Nombre de la huella (carpeta y ficheros de resultados).\n"
             "Se ha sacado de los ficheros quitando el numero de parte; cambialo si quieres:")
    if MAC:
        r = _osa(OSA_NOMBRE, texto, propuesto)
    else:
        from tkinter import simpledialog
        r = simpledialog.askstring("Nombre de la huella", texto, initialvalue=propuesto, parent=_tk_raiz())
    return r.strip() if r and r.strip() else None


OSA_TEXTO = """on run argv
    activate
    return text returned of (display dialog (item 1 of argv) default answer (item 2 of argv) with title (item 3 of argv))
end run"""


def pedir_carga(propuesta):
    texto = ("Carga normal del ensayo Fn (N), para la tasa de desgaste especifica k.\n"
             + ("Leida del nombre del fichero/carpeta; corrigela si no es correcta:" if propuesta
                else "No se ha encontrado en el nombre. Escribela (deja vacio para no calcular k):"))
    defecto = f"{propuesta:g}" if propuesta else ""
    if MAC:
        r = _osa(OSA_TEXTO, texto, defecto, "Carga del ensayo")
    else:
        from tkinter import simpledialog
        r = simpledialog.askstring("Carga del ensayo", texto, initialvalue=defecto, parent=_tk_raiz())
    try:
        return float(r.replace(",", ".")) if r and r.strip() else None
    except ValueError:
        return None


def pedir_numero(titulo, texto):
    """Pide un numero en una ventana; vacio o cancelar -> None."""
    if MAC:
        r = _osa(OSA_TEXTO, texto, "", titulo)
    else:
        from tkinter import simpledialog
        r = simpledialog.askstring(titulo, texto, parent=_tk_raiz())
    try:
        return float(r.replace(",", ".")) if r and r.strip() else None
    except ValueError:
        return None


def pedir_pixel():
    texto = "Los ficheros no traen tamano de pixel.\nIntroduce el tamano de pixel (um):"
    if MAC:
        r = _osa(OSA_PIXEL, texto)
        try:
            return float(r.replace(",", ".")) if r else None
        except ValueError:
            return None
    from tkinter import simpledialog
    return simpledialog.askfloat("Tamano de pixel", texto, parent=_tk_raiz(), minvalue=1e-4, maxvalue=100)


BARRIDOS = (("Izquierda → derecha", "x"), ("Arriba → abajo", "y"), ("Automatico", "auto"))


def pedir_barrido():
    """En que direccion se han ido moviendo los campos a lo largo de la huella."""
    texto = ("¿Como has ido moviendo los campos a lo largo de la huella?\n\n"
             "Izquierda → derecha: huella horizontal\n"
             "Arriba → abajo: huella vertical\n"
             "Automatico: no lo se / campos en rejilla")
    if MAC:
        return dict(BARRIDOS).get(_osa(OSA_BARRIDO, texto, *(t for t, _ in BARRIDOS)))
    from tkinter import messagebox
    r = messagebox.askyesnocancel("Direccion del barrido",
                                  texto + "\n\nSi = izquierda → derecha   No = arriba → abajo   "
                                          "Cancelar = automatico", parent=_tk_raiz())
    return "auto" if r is None else ("x" if r else "y")


OSA_SINO = """on run argv
    activate
    return button returned of (display dialog (item 1 of argv) with title "Campos que no encajan" buttons {"Cancelar", "Coser igualmente"} default button 1 cancel button 1 with icon caution)
end run"""


def preguntar_forzar(texto):
    if MAC:
        return _osa(OSA_SINO, texto) == "Coser igualmente"
    from tkinter import messagebox
    return messagebox.askyesno("Campos que no encajan", texto, icon="warning", parent=_tk_raiz())


def aviso_gui(titulo, texto, error=False):
    if MAC:
        _osa(OSA_AVISO, titulo, texto, "error" if error else "info")
        return
    from tkinter import messagebox
    (messagebox.showerror if error else messagebox.showinfo)(titulo, texto, parent=_tk_raiz())


def main():
    ap = argparse.ArgumentParser(description="Stitching de topografias Sensofar medidas a mano")
    ap.add_argument("entradas", nargs="*",
                    help="carpeta con los campos o lista de ficheros (sin nada: ventana de seleccion)")
    ap.add_argument("--patron", default=None)
    ap.add_argument("--pixel", type=float, default=None, help="tamano de pixel (um)")
    ap.add_argument("--ref-frac", type=float, default=0.15)
    ap.add_argument("--ref-eje", choices=["auto", "x", "y"], default="auto")
    ap.add_argument("--umbral", type=float, default=3.0,
                    help="umbral de volumen en multiplos de sigma de la referencia")
    ap.add_argument("--tilt", action="store_true")
    ap.add_argument("--sin-filtro", action="store_true", help="no quitar picos aislados")
    ap.add_argument("--forzar", action="store_true",
                    help="cose aunque la correlacion sea baja (no recomendado)")
    ap.add_argument("--barrido", choices=["auto", "x", "y"], default="auto",
                    help="direccion en la que avanzan los campos: x = izquierda/derecha, "
                         "y = arriba/abajo, auto = cualquiera")
    ap.add_argument("--sin-3d", action="store_true",
                    help="no abrir el visor 3D en el navegador al terminar")
    ap.add_argument("--carga", type=float, default=None,
                    help="carga normal Fn (N); por defecto se lee del nombre (p.ej. '1N', 'CARGA 0.5N')")
    ap.add_argument("--distancia", type=float, default=None,
                    help="distancia total deslizada d (m) para k (sin ella no se calcula k)")
    ap.add_argument("--carrera", type=float, default=None,
                    help="carrera / longitud de la huella (um) para k (sin ella no se calcula k)")
    ap.add_argument("--nombre", default=None,
                    help="nombre de la huella (por defecto, el de los ficheros sin el numero de parte)")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    a.gui = not a.entradas
    global MODO_GUI
    MODO_GUI = a.gui
    if a.gui:
        a.entradas = elegir_ficheros()
        if not a.entradas:
            sys.exit("Cancelado: no se selecciono ningun fichero.")
        if len(a.entradas) > 1:
            a.barrido = pedir_barrido()
            if a.barrido is None:
                sys.exit("Cancelado: no se eligio la direccion del barrido.")
    # al barrer a lo largo de la huella, la referencia sin desgaste esta a los lados
    if a.ref_eje == "auto" and a.barrido != "auto":
        a.ref_eje = "y" if a.barrido == "x" else "x"
    return procesar(a)


def procesar(a):
    ficheros = []
    for e in a.entradas:
        if os.path.isdir(e):
            if a.patron:
                ficheros += glob.glob(os.path.join(e, a.patron))
            else:
                ficheros += [f for f in glob.glob(os.path.join(e, "*"))
                             if os.path.splitext(f)[1].lower() in EXTENSIONES]
        elif os.path.isfile(e):
            ficheros.append(e)
        else:
            sys.exit(f"ERROR: no existe {e}")
    ficheros = sorted(set(ficheros), key=lambda f: orden_natural(os.path.basename(f)))
    if not ficheros:
        sys.exit("ERROR: no hay campos que coser.")
    a.carpeta = a.entradas[0] if os.path.isdir(a.entradas[0]) else os.path.dirname(os.path.abspath(ficheros[0]))
    nombre = a.nombre or nombre_huella(ficheros)
    if a.gui and not a.nombre:
        nombre = pedir_nombre(nombre)
        if nombre is None:
            sys.exit("Cancelado: no se indico el nombre de la huella.")
    nombre = re.sub(r'[/\\:*?"<>|]', "_", nombre).strip() or "huella"
    if a.carga is None:
        a.carga = carga_de_ruta(ficheros[0])
        if a.gui:
            a.carga = pedir_carga(a.carga)
    if a.gui and a.carga:
        if a.distancia is None:
            a.distancia = pedir_numero("Distancia deslizada",
                                       "Distancia total deslizada en el ensayo d (m), para k\n"
                                       "(deja vacio para no calcular k):")
        if a.distancia and a.carrera is None:
            a.carrera = pedir_numero("Carrera", "Carrera / longitud de la huella (um), para k\n"
                                                "(deja vacio para no calcular k):")
    outdir = a.salida or os.path.join(a.carpeta, nombre)
    os.makedirs(outdir, exist_ok=True)

    def ruta(fichero):
        return os.path.join(outdir, f"{nombre}_{fichero}")

    print(f"Leyendo {len(ficheros)} campos...")
    campos = []
    for f in ficheros:
        if a.gui and a.pixel is None and os.path.splitext(f)[1].lower() not in (".plu", ".gsf"):
            # matriz ASCII/npy sin pixel: se pregunta una sola vez en la ventana
            z0, px0 = (np.load(f).astype(float), None) if f.lower().endswith(".npy") else leer_ascii(f)
            if px0 is None:
                a.pixel = pedir_pixel()
                if a.pixel is None:
                    sys.exit("Cancelado: hace falta el tamano de pixel.")
        z, px = leer_campo(f, a.pixel)
        npicos = 0
        if not a.sin_filtro:
            z, npicos = quitar_picos(z)
        nan = np.isnan(z).mean() * 100
        print(f"  {os.path.basename(f)}: {z.shape[1]}x{z.shape[0]} px, pixel {px:.4f} um, "
              f"no medidos {nan:.1f}% (picos quitados: {npicos})")
        campos.append((z, px))
    pxs = {round(p, 6) for _, p in campos}
    if len(pxs) > 1:
        sys.exit(f"ERROR: los campos tienen distinto tamano de pixel {pxs}. Usa el mismo objetivo.")
    px = campos[0][1]

    if len(campos) == 1:
        print("Un solo campo: sin cosido (solo nivelado, volumen y 3D).")
        pos, calidad = [np.array([0, 0])], [(None, 1.0)]
    else:
        print("Alineando en XY...")
        pos, calidad = registrar(campos, barrido=a.barrido, forzar=a.forzar,
                                 preguntar=preguntar_forzar if a.gui else None)
    print("Ajustando Z entre campos...")
    zs, corr = ajustar_z(campos, pos, con_tilt=a.tilt)
    mos, pos_rel = componer(zs, pos)

    ref, eje = mascara_referencia(mos.shape, a.ref_frac, a.ref_eje, mos)
    niv, sigma_ref = nivelar(mos, ref)
    vol = volumenes(niv, ref, px, sigma_ref, a.umbral)
    s, A, valid = area_seccion(niv, ref, px, eje, sigma_ref, a.umbral)
    _, A0, _ = area_seccion(niv, ref, px, eje, sigma_ref, 0.0)   # sin umbral (todo lo bajo el plano)
    con_k = all(v is not None and v > 0 for v in (a.carga, a.distancia, a.carrera))
    kA = k_archard(A, a.carrera, a.carga, a.distancia) if con_k else np.full_like(A, np.nan)
    kA0 = k_archard(A0, a.carrera, a.carga, a.distancia) if con_k else np.full_like(A, np.nan)

    # salidas
    np.save(ruta("mosaico.npy"), niv)
    guardar_gsf(ruta("mosaico.gsf"), np.nan_to_num(niv, nan=0.0), px)
    np.savetxt(ruta("mosaico_matriz.txt"), niv, fmt="%.4f",
               header=f"alturas en um; pixel = {px} um; {niv.shape[1]} columnas x {niv.shape[0]} filas")
    np.savetxt(ruta("area_seccion.csv"),
               np.c_[s[valid], A[valid], A0[valid], kA[valid], kA0[valid]], delimiter=",", fmt="%.4f",
               header="posicion_um,area_umbral_um2,area_sin_umbral_um2,"
                      "k_umbral_1e-6mm3_Nm,k_sin_umbral_1e-6mm3_Nm", comments="")
    figuras(niv, pos_rel, [z.shape for z in zs], px, eje, ref, s, A, valid, ruta)

    sel = valid & (A > 0.1 * A[valid].max()) if valid.any() and A[valid].max() > 0 else np.zeros_like(valid)
    Av, A0v = A[sel], A0[sel]

    def media_sd(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else float("nan"))

    if con_k and Av.size:
        k_m, k_sd = media_sd(kA[sel])
        k0_m, k0_sd = media_sd(kA0[sel])
        bloque_k = [
            f"  Fn = {a.carga:g} N   d = {a.distancia:g} m   carrera = {a.carrera:g} um",
            f"  k (area con umbral)  = {k_m:.3f} ± {k_sd:.3f} x10^-6 mm3/(N·m)",
            f"  k (area sin umbral)  = {k0_m:.3f} ± {k0_sd:.3f} x10^-6 mm3/(N·m)",
            "  k = Area_seccion x carrera / (Fn x d);",
            "  media ± desv. estandar de las secciones con desgaste a lo largo del mosaico.",
        ]
    else:
        k_m = k_sd = k0_m = k0_sd = float("nan")
        bloque_k = ["  no calculada: " + ("faltan carga, distancia o carrera (--carga, --distancia, --carrera)"
                                          if not con_k else "no hay secciones con desgaste")]
    # resumen de una fila para pegar en el Excel
    with open(ruta("resumen.csv"), "w", encoding="utf-8") as fr:
        fr.write("huella,Fn_N,d_m,carrera_um,area_umbral_media_um2,area_umbral_sd_um2,"
                 "area_sin_umbral_media_um2,area_sin_umbral_sd_um2,k_umbral_1e-6mm3_Nm,k_umbral_sd,"
                 "k_sin_umbral_1e-6mm3_Nm,k_sin_umbral_sd,n_secciones,prof_max_um\n")
        am, asd = media_sd(Av) if Av.size else (float("nan"),) * 2
        a0m, a0sd = media_sd(A0v) if A0v.size else (float("nan"),) * 2
        fr.write(f"{nombre},{a.carga if con_k else ''},{a.distancia or ''},{a.carrera or ''},"
                 f"{am:.2f},{asd:.2f},{a0m:.2f},{a0sd:.2f},{k_m:.4f},{k_sd:.4f},"
                 f"{k0_m:.4f},{k0_sd:.4f},{Av.size},{vol['prof_max']:.3f}\n")
    lineas = [
        f"INFORME DE STITCHING - {nombre}", "=" * 60,
        f"Carpeta: {os.path.abspath(a.carpeta)}",
        f"Campos: {len(ficheros)}   pixel: {px:.4f} um   barrido: {a.barrido}",
        f"Mosaico: {niv.shape[1]} x {niv.shape[0]} px = {niv.shape[1]*px/1000:.3f} x {niv.shape[0]*px/1000:.3f} mm",
        f"Puntos no medidos en el mosaico: {np.isnan(niv).mean()*100:.1f} %",
        "", "Posicion y calidad de cada campo:",
        "  campo  fichero                      x(px)   y(px)  ref  NCC    dZ(um)",
    ]
    for k, f in enumerate(ficheros):
        j, ncc = calidad[k]
        lineas.append(f"  {k+1:4d}   {os.path.basename(f)[:26]:26s} {pos_rel[k][1]:7d} {pos_rel[k][0]:7d}"
                      f"  {'-' if j is None else j+1:>3}  {ncc:5.3f}  {corr[k][0]:+8.3f}")
    lineas += [
        f"  (NCC < 0.5 => revisa esa union en {nombre}_mosaico.png)",
        "",
        f"Referencia: bandas del {a.ref_frac*100:.0f} % en los bordes "
        f"{'superior/inferior' if eje=='y' else 'izquierdo/derecho'} (huella a lo largo de {'X' if eje=='y' else 'Y'})",
        f"Sigma de la superficie de referencia: {sigma_ref:.4f} um",
        "",
        "VOLUMENES (fuera de las bandas de referencia):",
        f"  V- total (bajo el plano)            : {vol['V_menos']:.4e} um3 = {vol['V_menos']*1e-9:.5f} mm3",
        f"  V- con umbral {a.umbral:.0f} sigma ({vol['umbral']:.3f} um)  : {vol['V_menos_umbral']:.4e} um3 = {vol['V_menos_umbral']*1e-9:.5f} mm3",
        f"  V+ total (material apilado)         : {vol['V_mas']:.4e} um3",
        f"  V+ con umbral                       : {vol['V_mas_umbral']:.4e} um3",
        f"  Profundidad maxima                  : {vol['prof_max']:.3f} um",
        "",
        "AREA DE SECCION, a lo largo de la huella:",
        (f"  con umbral {a.umbral:g} sigma : media = {Av.mean():.1f} um2   desv. = {Av.std(ddof=1) if Av.size > 1 else 0:.1f} um2"
         f"   (n = {Av.size} secciones, ~ {Av.size*px/1000:.3f} mm)"
         if Av.size else "  sin secciones con desgaste por encima del umbral"),
        (f"  sin umbral        : media = {A0v.mean():.1f} um2   desv. = {A0v.std(ddof=1) if A0v.size > 1 else 0:.1f} um2"
         if A0v.size else ""),
        "",
        "TASA DE DESGASTE ESPECIFICA (Archard):",
        *bloque_k,
        "",
        "Notas:",
        "  * Usa V- con umbral como volumen de desgaste (equivalente a quitar el ruido de la rugosidad).",
        "  * Comprueba que las bandas de referencia (rojo en mosaico.png) caen sobre superficie sin desgastar;",
        "    si no, cambia --ref-frac o --ref-eje.",
    ]
    txt = "\n".join(lineas)
    with open(ruta("informe.txt"), "w", encoding="utf-8") as fi:
        fi.write(txt)
    print()
    print(txt)
    print(f"\nResultados en: {os.path.abspath(outdir)}")
    html = ruta("mosaico_3d.html")
    if not a.sin_3d and os.path.exists(html):
        import webbrowser
        print("Abriendo el visor 3D en el navegador...")
        from pathlib import Path
        webbrowser.open(Path(html).resolve().as_uri())
    if a.gui:
        res_k = (f"Tasa de desgaste especifica (Fn = {a.carga:g} N), x10⁻⁶ mm³/(N·m):\n"
                 f"  sin umbral: k = {k0_m:.2f} ± {k0_sd:.2f}   (A = {A0v.mean():.0f} µm²)\n"
                 f"  con umbral: k = {k_m:.2f} ± {k_sd:.2f}   (A = {Av.mean():.0f} µm²)\n\n"
                 if con_k and Av.size else "")
        aviso_gui("Cosido terminado", res_k + f"Resultados en:\n{os.path.abspath(outdir)}\n\n"
                  "El modelo 3D se ha abierto en el navegador (mosaico_3d.html).")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if MODO_GUI and isinstance(e.code, str) and not e.code.startswith("Cancelado"):
            aviso_gui("Error", e.code, error=True)
        raise
    finally:
        cerrar_gui()
