#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exportador a OriginLab — ensayos tribologicos reciprocantes (Block-on-Flat)

Lee los ficheros binarios .tst de un tribometro (formato CETR/UMT), calcula
CoF, fuerzas, desgaste vertical Z1 y estadisticos (modulo bof_core.py) y
genera salidas pensadas para importar directamente en OriginLab:

  1.  <id>_origin.dat    — Serie temporal completa en texto delimitado con
                           cabecera de 3 filas (Long Name / Units / Comments).
                           Origin la reconoce automáticamente al importar
                           (ASCII / Import Wizard / Data Connector).

  2.  <id>_origin_dec.dat— Igual pero diezmada (si el factor de diezmado > 1),
                           práctica para trazar rápido sin cargar ~10^5 filas.

  3.  resumen_origin.dat — Una fila de KPIs por ensayo (Origin-ready, 3 filas
                           de cabecera). Se acumula entre ejecuciones usando
                           `resumen_origin.json` como fuente de verdad: volver
                           a exportar un ensayo actualiza su fila, no la duplica.

  4.  <id>.opju         — UN proyecto Origin POR CADA ensayo, guardado
                           junto a su .tst con el mismo nombre (igual que
                           <id>.html). Contiene la hoja de datos del ensayo
                           (+ gráficas opcionales de CoF, CoF suavizado
                           y Z1).
                           Requiere el paquete `originpro` y OriginLab
                           instalado (>= 2021). Si no está disponible se
                           OMITE con un aviso, sin abortar el resto.

Uso: ejecutar y seleccionar los .tst en el diálogo. Los ultimos valores
usados se recuerdan en ~/.exportar_origin_config.json.

Los ficheros .dat/.csv se generan en Windows, macOS y Linux. El proyecto
.opju requiere OriginLab, que solo existe para Windows.
"""

# ── AUTO-INSTALL ──────────────────────────────────────────────────────────────
import subprocess, sys

# Consolas de Windows (cp1252): que un print con µ, ±, → nunca aborte el script
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

def _pip_install(pkg):
    # venvs creados con uv no traen pip: usar uv si pip no está disponible
    import shutil
    if subprocess.call([sys.executable, "-m", "pip", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    elif shutil.which("uv"):
        subprocess.check_call(["uv", "pip", "install", "--python", sys.executable, pkg, "-q"])
    else:
        raise RuntimeError(f"No hay pip ni uv para instalar {pkg}")

for _pkg, _mod in [("numpy", "numpy"), ("pandas", "pandas")]:
    try:
        __import__(_mod)
    except ImportError:
        print(f"Instalando {_pkg}...")
        _pip_install(_pkg)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
except ImportError:
    sys.exit("ERROR: falta tkinter (interfaz grafica). En Debian/Ubuntu: sudo apt install python3-tk")

# ── COLORES ───────────────────────────────────────────────────────────────────
CA = "#1A3A5C"; CO = "#D4570A"; CP = "#7B2D8B"

# Fuente de la interfaz segun el sistema (Segoe UI solo existe en Windows)
FONT = {"win32": "Segoe UI", "darwin": "Helvetica Neue"}.get(sys.platform, "DejaVu Sans")

CONFIG_FILE = Path.home() / ".exportar_origin_config.json"


# Motor de calculo (lector .tst + estadisticos), en el mismo directorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bof_core  # noqa: E402


def _mod_bof():
    return bof_core


# ═════════════════════════════════════════════════════════════════════════════
# ESPECIFICACIÓN DE COLUMNAS
# ═════════════════════════════════════════════════════════════════════════════
#   (columna_df,   Long Name,             Units,  Comment)
COLS_SERIE = [
    ("tiempo_s",     "Tiempo",             "s",    ""),
    ("tiempo_min",   "Tiempo",             "min",  ""),
    ("distancia_m",  "Distancia",          "m",    "d = v_media * t"),
    ("Fz_N",         "Fz",                 "N",    "Carga normal (magnitud)"),
    ("Fx_N",         "Fx",                 "N",    "Fuerza de friccion (con signo, alternante)"),
    ("Fx_abs",       "|Fx|",               "N",    "Modulo de Fx"),
    ("Z1_mm",        "Z1",                 "mm",   "Penetracion / desgaste vertical"),
    ("CoF",          "CoF",                "",     "|Fx| / Fz instantaneo"),
    ("CoF_roll",     "CoF (media movil)",  "",     "suavizado ventana configurada"),
    ("Fx_roll",      "|Fx| (media movil)", "N",    "suavizado ventana configurada"),
    ("Z1_roll",      "Z1 (media movil)",   "mm",   "suavizado ventana configurada"),
]

#   (clave,           Long Name,               Units)
COLS_RESUMEN = [
    ("id_ensayo",       "ID Ensayo",             ""),
    ("lote",            "Lote / muestra",        ""),
    ("zona",            "Zona",                  ""),
    ("tratamiento",     "Tratamiento",           ""),
    ("carga_N",         "Carga nominal",         "N"),
    ("velocidad_ms",    "Velocidad media",       "m/s"),
    ("frecuencia_hz",   "Frecuencia",            "Hz"),
    ("duracion_min",    "Duracion",              "min"),
    ("distancia_m",     "Distancia",             "m"),
    ("n_ciclos",        "N ciclos",              ""),
    ("cof_est_med",     "CoF estacionario",      ""),
    ("cof_est_sd",      "CoF estac. SD",         ""),
    ("cof_med",         "CoF medio total",       ""),
    ("cof_sd",          "CoF SD total",          ""),
    ("cof_min",         "CoF min (suav.)",       ""),
    ("cof_max",         "CoF max (suav.)",       ""),
    ("cof_t1",          "CoF tercio 1",          ""),
    ("cof_t2",          "CoF tercio 2",          ""),
    ("cof_t3",          "CoF tercio 3",          ""),
    ("fz_med",          "Fz media",              "N"),
    ("fz_sd",           "Fz SD",                 "N"),
    ("fx_med",          "|Fx| media",            "N"),
    ("fx_sd",           "|Fx| SD",               "N"),
    ("z1_init",         "Z1 inicial",            "mm"),
    ("z1_final",        "Z1 final",              "mm"),
    ("z1_range",        "dZ1 (desgaste)",        "mm"),
    ("z1_rate_nm_m",    "Tasa desgaste Z1",      "nm/m"),
    ("dist_est_m",      "Dist. inicio estac.",   "m"),
    ("d_runin_m",       "Fin de rodadura",       "m"),
    ("hz_adq",          "Frec. adquisicion",     "Hz"),
    ("n_muestras",      "N muestras",            ""),
    ("ventana_s",       "Ventana suavizado",     "s"),
    ("pct_estacionario","Fraccion estacionario", ""),
    ("fecha_analisis",  "Fecha analisis",        ""),
    ("archivo_tst",     "Archivo .tst",          ""),
]


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG PERSISTENTE
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# DIÁLOGO
# ═════════════════════════════════════════════════════════════════════════════

class OriginDialog:
    _FIELDS = [
        ("lote",             "Lote / muestra:",              str,   24),
        ("tratamiento",      "Tratamiento:",                 str,   30),
        ("zona",             "Zona:",                        str,   18),
        ("carga_N",          "Carga nominal (N):",           float, 10),
        ("velocidad_ms",     "Velocidad media (m/s):",       float, 10),
        ("frecuencia_hz",    "Frecuencia (Hz):",             float, 10),
        ("ventana_s",        "Ventana suavizado (s):",       float, 10),
        ("pct_estacionario", "Fracción estacionario (0-1):", float, 10),
    ]

    def __init__(self):
        bof = _mod_bof()
        cfg      = _load_json(CONFIG_FILE)

        defaults = {**bof.PARAM_DEFAULTS, **cfg.get("params", {})}
        self._saved     = defaults
        self._last_dir  = cfg.get("last_dir", str(Path.home()))
        opt = cfg.get("origin", {})
        self._o_decim    = opt.get("decimado", 1)
        self._o_delim    = opt.get("delimitador", "tab")
        self._o_opju     = opt.get("opju", True)
        self._o_graphs   = opt.get("graficas", True)
        self._o_summary  = opt.get("resumen_path",
                            str(Path.home() / "Documents" / "resumen_ensayos_origin.dat"))

        self.cancelled = True
        self.files:  list[tuple[str, str]] = []
        self.params: dict = {}
        self.opts:   dict = {}
        self._run()

    # ── UI ────────────────────────────────────────────────────────
    def _run(self):
        r = tk.Tk()
        self._root = r
        r.title("Exportar a OriginLab — BOF")
        r.configure(bg="#F2F4F7")

        style = ttk.Style(r)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabelframe", background="#F2F4F7")
        style.configure("TLabelframe.Label", background="#F2F4F7",
                        foreground=CA, font=(FONT, 9, "bold"))
        style.configure("TCheckbutton", background="#F2F4F7", font=(FONT, 9))
        style.configure("TRadiobutton", background="#F2F4F7", font=(FONT, 9))

        banner = tk.Frame(r, bg=CA, pady=10)
        banner.pack(fill="x")
        tk.Label(banner, text="Exportar a OriginLab — Block-on-Flat",
                 font=(FONT, 14, "bold"), bg=CA, fg="white").pack()
        tk.Label(banner, text="Lector .tst  ·  serie temporal + resumen de KPIs + proyecto .opju",
                 font=(FONT, 8), bg=CA, fg="#8BAFD0").pack()

        main = tk.Frame(r, bg="#F2F4F7", padx=12, pady=8)
        main.pack(fill="both", expand=True)

        # ── Archivos ──────────────────────────────────────────────
        f_files = ttk.LabelFrame(main, text=" Archivos .tst ", padding=6)
        f_files.pack(fill="x", pady=(0, 6))

        tw = tk.Frame(f_files, bg="#F2F4F7")
        tw.pack(side="left", fill="both", expand=True)
        self._tree = ttk.Treeview(tw, columns=("path", "id"), show="headings",
                                  height=4, selectmode="browse")
        self._tree.heading("path", text="Archivo .tst")
        self._tree.heading("id",   text="ID Ensayo  (doble-clic para editar)")
        self._tree.column("path", width=340, stretch=True)
        self._tree.column("id",   width=190, stretch=True)
        vsb = ttk.Scrollbar(tw, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        self._tree.bind("<Double-1>", self._inline_edit)

        bc = tk.Frame(f_files, bg="#F2F4F7")
        bc.pack(side="left", padx=(8, 0), anchor="n")
        ttk.Button(bc, text="Añadir…", command=self._add_files ).pack(fill="x", pady=2)
        ttk.Button(bc, text="Quitar",  command=self._remove_file).pack(fill="x", pady=2)

        # ── Metadatos ─────────────────────────────────────────────
        f_meta = ttk.LabelFrame(main, text=" Metadatos del ensayo ", padding=8)
        f_meta.pack(fill="x", pady=(0, 6))
        self._vars: dict[str, tuple[tk.StringVar, type]] = {}
        for idx, (key, label, typ, w) in enumerate(self._FIELDS):
            row, cb = divmod(idx, 2)
            cb *= 2
            tk.Label(f_meta, text=label, bg="#F2F4F7", font=(FONT, 9),
                     anchor="e").grid(row=row, column=cb, sticky="e",
                                      padx=(8 if cb else 0, 4), pady=3)
            var = tk.StringVar(value=str(self._saved.get(key, "")))
            self._vars[key] = (var, typ)
            ttk.Entry(f_meta, textvariable=var, width=w).grid(
                row=row, column=cb + 1, sticky="ew", padx=(0, 10), pady=3)
        f_meta.columnconfigure(1, weight=1)
        f_meta.columnconfigure(3, weight=1)

        # ── Opciones Origin ───────────────────────────────────────
        f_o = ttk.LabelFrame(main, text=" Opciones de exportación Origin ", padding=8)
        f_o.pack(fill="x", pady=(0, 6))

        r1 = tk.Frame(f_o, bg="#F2F4F7"); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Delimitador:", bg="#F2F4F7",
                 font=(FONT, 9)).pack(side="left")
        self._v_delim = tk.StringVar(value=self._o_delim)
        ttk.Radiobutton(r1, text="Tabulador", value="tab",
                        variable=self._v_delim).pack(side="left", padx=6)
        ttk.Radiobutton(r1, text="Coma (CSV)", value="coma",
                        variable=self._v_delim).pack(side="left", padx=6)
        ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=10)
        tk.Label(r1, text="Diezmado (1 = todos los puntos):", bg="#F2F4F7",
                 font=(FONT, 9)).pack(side="left")
        self._v_decim = tk.StringVar(value=str(self._o_decim))
        ttk.Entry(r1, textvariable=self._v_decim, width=6).pack(side="left", padx=6)

        r2 = tk.Frame(f_o, bg="#F2F4F7"); r2.pack(fill="x", pady=2)
        self._v_opju   = tk.BooleanVar(value=self._o_opju)
        self._v_graphs = tk.BooleanVar(value=self._o_graphs)
        ttk.Checkbutton(r2, text="Generar proyecto Origin (.opju) — requiere 'originpro' + OriginLab",
                        variable=self._v_opju).pack(side="left")
        ttk.Checkbutton(r2, text="  con gráficas",
                        variable=self._v_graphs).pack(side="left", padx=10)

        r3 = tk.Frame(f_o, bg="#F2F4F7"); r3.pack(fill="x", pady=(4, 0))
        tk.Label(r3, text="Resumen KPIs (.dat):", bg="#F2F4F7",
                 font=(FONT, 9)).pack(side="left")
        self._v_summary = tk.StringVar(value=self._o_summary)
        ttk.Entry(r3, textvariable=self._v_summary).pack(
            side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(r3, text="…", width=3, command=self._pick_summary).pack(side="left")

        # ── Botones ───────────────────────────────────────────────
        bb = tk.Frame(r, bg="#E8EAF0", pady=8)
        bb.pack(fill="x")
        ttk.Button(bb, text="Cancelar",     command=self._cancel).pack(side="right", padx=8)
        ttk.Button(bb, text="▶  Exportar",  command=self._ok    ).pack(side="right", padx=4)

        r.update_idletasks()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h   = r.winfo_reqwidth(),    r.winfo_reqheight()
        r.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
        r.mainloop()

    # ── acciones ──────────────────────────────────────────────────
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona archivos .tst", initialdir=self._last_dir,
            filetypes=[("Archivos TST", "*.tst"), ("Todos", "*.*")],
            parent=self._root)
        for p in paths:
            self._tree.insert("", "end", values=(p, Path(p).stem))
            self._last_dir = str(Path(p).parent)

    def _remove_file(self):
        for item in self._tree.selection():
            self._tree.delete(item)

    def _pick_summary(self):
        p = filedialog.asksaveasfilename(
            title="Archivo resumen de KPIs para Origin",
            initialfile=Path(self._v_summary.get()).name,
            initialdir=str(Path(self._v_summary.get()).parent),
            defaultextension=".dat",
            filetypes=[("Datos Origin", "*.dat"), ("Texto", "*.txt"), ("CSV", "*.csv")],
            confirmoverwrite=False, parent=self._root)
        if p:
            self._v_summary.set(p)

    def _inline_edit(self, event):
        if self._tree.identify_column(event.x) != "#2":
            return
        item = self._tree.identify_row(event.y)
        if not item:
            return
        bbox = self._tree.bbox(item, "#2")
        if not bbox:
            return
        x, y, w, h = bbox
        var = tk.StringVar(value=self._tree.set(item, "id"))
        entry = ttk.Entry(self._tree, textvariable=var, font=(FONT, 9))
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set(); entry.select_range(0, "end")

        def commit(e=None):
            v = var.get().strip() or Path(self._tree.set(item, "path")).stem
            self._tree.set(item, "id", v)
            entry.destroy()

        entry.bind("<Return>",   commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>",   lambda e: entry.destroy())

    def _cancel(self):
        self.cancelled = True
        self._root.destroy()

    def _ok(self):
        items = self._tree.get_children()
        if not items:
            messagebox.showwarning("Sin archivos",
                                   "Añade al menos un archivo .tst.", parent=self._root)
            return
        self.files = [(self._tree.set(i, "path"), self._tree.set(i, "id")) for i in items]

        self.params = {}
        for key, (var, typ) in self._vars.items():
            try:
                self.params[key] = typ(var.get().strip())
            except ValueError:
                messagebox.showerror(
                    "Valor inválido",
                    f"Campo '{key}': '{var.get()}' no es válido ({typ.__name__}).",
                    parent=self._root)
                return

        try:
            decim = max(1, int(float(self._v_decim.get())))
        except ValueError:
            decim = 1

        self.opts = {
            "delimitador":  self._v_delim.get(),
            "decimado":     decim,
            "opju":         self._v_opju.get(),
            "graficas":     self._v_graphs.get(),
            "resumen_path": self._v_summary.get().strip(),
        }

        _save_json(CONFIG_FILE, {
            "last_dir": self._last_dir,
            "params":   self.params,
            "origin": {
                "delimitador":  self.opts["delimitador"],
                "decimado":     self.opts["decimado"],
                "opju":         self.opts["opju"],
                "graficas":     self.opts["graficas"],
                "resumen_path": self.opts["resumen_path"],
            },
        })
        self.cancelled = False
        self._root.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# ESCRITURA DE FICHEROS ORIGIN-READY (texto delimitado, cabecera de 3 filas)
# ═════════════════════════════════════════════════════════════════════════════

def _delim_char(nombre: str) -> str:
    return "," if nombre == "coma" else "\t"


def escribir_serie(df: pd.DataFrame, out_path: Path, sep: str):
    """Serie temporal en el formato que Origin autodetecta:
        fila 1 = Long Names, fila 2 = Units, fila 3 = Comments, fila 4+ = datos."""
    cols = [c for c in COLS_SERIE if c[0] in df.columns]
    names = [c[0] for c in cols]
    hdr_long  = sep.join(c[1] for c in cols)
    hdr_units = sep.join(c[2] for c in cols)
    hdr_comm  = sep.join(c[3] for c in cols)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr_long  + "\n")
        fh.write(hdr_units + "\n")
        fh.write(hdr_comm  + "\n")
        df[names].to_csv(fh, sep=sep, index=False, header=False,
                         float_format="%.6f")


def _resumen_dict(P: dict, stats: dict) -> dict:
    s = stats
    return {
        "id_ensayo":        P["id_ensayo"],
        "lote":             P.get("lote", ""),
        "zona":             P.get("zona", ""),
        "tratamiento":      P.get("tratamiento", ""),
        "carga_N":          P["carga_N"],
        "velocidad_ms":     P["velocidad_ms"],
        "frecuencia_hz":    P["frecuencia_hz"],
        "duracion_min":     s["t_max"] / 60.0,
        "distancia_m":      s["d_max"],
        "n_ciclos":         s["n_ciclos"],
        "cof_est_med":      s["cof_est_med"],
        "cof_est_sd":       s["cof_est_sd"],
        "cof_med":          s["cof_med"],
        "cof_sd":           s["cof_sd"],
        "cof_min":          s["cof_min"],
        "cof_max":          s["cof_max"],
        "cof_t1":           s["cof_tercios"][0],
        "cof_t2":           s["cof_tercios"][1],
        "cof_t3":           s["cof_tercios"][2],
        "fz_med":           s["fz_med"],
        "fz_sd":            s["fz_sd"],
        "fx_med":           s["fx_med"],
        "fx_sd":            s["fx_sd"],
        "z1_init":          s["z1_init"],
        "z1_final":         s["z1_final"],
        "z1_range":         s["z1_range"],
        "z1_rate_nm_m":     s["z1_rate"] * 1e6,
        "dist_est_m":       s["dist_est"],
        "d_runin_m":        s["d_runin"] if s["d_runin"] is not None else "",
        "hz_adq":           s["hz_adq"],
        "n_muestras":       s["n_muestras"],
        "ventana_s":        P["ventana_s"],
        "pct_estacionario": P["pct_estacionario"],
        "fecha_analisis":   date.today().isoformat(),
        "archivo_tst":      str(P["tst_file"]),
    }


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:            # NaN
            return ""
        return f"{v:.6g}"
    return str(v)


def escribir_resumen(rows: list[dict], out_path: Path, sep: str):
    """Escribe el .dat de KPIs (cabecera de 3 filas). `rows` ya viene
    consolidado (sin duplicados de ID)."""
    keys = [c[0] for c in COLS_RESUMEN]
    hdr_long  = sep.join(c[1] for c in COLS_RESUMEN)
    hdr_units = sep.join(c[2] for c in COLS_RESUMEN)
    hdr_comm  = sep.join("" for _ in COLS_RESUMEN)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr_long  + "\n")
        fh.write(hdr_units + "\n")
        fh.write(hdr_comm  + "\n")
        for row in rows:
            fh.write(sep.join(_fmt_val(row.get(k, "")) for k in keys) + "\n")


def consolidar_resumen(nuevos: list[dict], base_dir: Path,
                       summary_path: Path, sep: str) -> Path:
    """Fusiona `nuevos` con el histórico (resumen_origin.json junto al .dat).
    Reexportar un ID actualiza su fila en vez de duplicarla. Devuelve la ruta
    del .dat escrito."""
    summary_path = summary_path if summary_path else (base_dir / "resumen_origin.dat")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = summary_path.with_suffix(".json")

    hist = _load_json(json_path)
    filas = hist.get("filas", []) if isinstance(hist, dict) else []

    por_id = {f["id_ensayo"]: f for f in filas if "id_ensayo" in f}
    for row in nuevos:
        por_id[row["id_ensayo"]] = row
    filas = list(por_id.values())

    _save_json(json_path, {"actualizado": date.today().isoformat(), "filas": filas})
    escribir_resumen(filas, summary_path, sep)
    return summary_path


# ═════════════════════════════════════════════════════════════════════════════
# PROYECTO ORIGIN (.opju)  — best-effort, requiere `originpro` + OriginLab
# ═════════════════════════════════════════════════════════════════════════════

def _sanitizar_nombre(s: str) -> str:
    bad = '[]{}():;,/\\?*"<>|\t\n'
    out = "".join("_" if ch in bad else ch for ch in str(s)).strip()
    return (out or "Ensayo")[:30]


def exportar_opju(series: list[tuple[dict, pd.DataFrame]],
                  filas_resumen: list[dict],
                  con_graficas: bool) -> list[Path]:
    """Genera UN proyecto Origin POR CADA ensayo, guardado junto a su .tst
    con el mismo nombre:  <id_ensayo>.opju  (igual que <id_ensayo>.html).
    series = [(P, df), ...] con df ya calculado. Devuelve la lista de .opju
    escritos."""
    try:
        import originpro as op
    except ImportError:
        print("  .opju OMITIDO: falta el paquete 'originpro'.")
        print("     Activalo con:  pip install originpro   (requiere OriginLab >= 2021, solo Windows)")
        return []

    try:
        op.set_show(False)
    except Exception:
        pass

    guardados: list[Path] = []
    try:
        for P, df in series:
            op.new()                       # proyecto limpio para este ensayo

            cols = [c for c in COLS_SERIE if c[0] in df.columns]
            sub = df[[c[0] for c in cols]].copy()
            sub.columns = [c[1] for c in cols]

            name = _sanitizar_nombre(P["id_ensayo"])
            wbd = op.new_book("w", lname=name)
            wsd = wbd[0]
            wsd.from_df(sub)
            try:
                wsd.set_labels([c[1] for c in cols], "L")
                wsd.set_labels([c[2] for c in cols], "U")
                wsd.set_labels([c[3] for c in cols], "C")
            except Exception:
                pass

            if con_graficas:
                _opju_graficas(op, wsd, cols, P)

            opju_path = Path(P["tst_file"]).parent / f"{P['id_ensayo']}.opju"
            opju_path.parent.mkdir(parents=True, exist_ok=True)
            ok = op.save(str(opju_path))
            print(f"  .opju: {'guardado' if ok else 'FALLO al guardar'} -> {opju_path}")
            if ok:
                guardados.append(opju_path)

        return guardados

    except Exception as exc:
        print(f"  .opju OMITIDO por error de OriginLab/originpro: {exc}")
        return guardados
    finally:
        try:
            op.exit()
        except Exception:
            pass


def _opju_graficas(op, wsd, cols, P):
    """Gráficas por ensayo:
        1) CoF vs TIEMPO  — CoF instantáneo + media móvil superpuestos
        2) CoF suavizado vs tiempo — solo la media móvil
        3) Z1 vs distancia — media móvil
    Cada una protegida: un fallo no aborta la exportación."""
    names = [c[0] for c in cols]

    def _idx(col):
        return names.index(col) if col in names else None

    # (título, x_col, [y_cols], x_label, y_label)
    specs = [
        (f"{P['id_ensayo']} — CoF vs tiempo", "tiempo_min",
         ["CoF", "CoF_roll"], "Tiempo (min)", "Coeficiente de fricción (CoF)"),
        (f"{P['id_ensayo']} — CoF suavizado", "tiempo_min",
         ["CoF_roll"], "Tiempo (min)", "CoF (media móvil)"),
        (f"{P['id_ensayo']} — Z1 vs distancia", "distancia_m",
         ["Z1_roll"], "Distancia de deslizamiento (m)", "Z1 (mm)"),
    ]
    for titulo, xcol, ycols, xlab, ylab in specs:
        cx  = _idx(xcol)
        cys = [_idx(y) for y in ycols if _idx(y) is not None]
        if cx is None or not cys:
            continue
        try:
            gp = None
            for tpl in ("line", "origin"):
                try:
                    gp = op.new_graph(template=tpl)
                    break
                except Exception:
                    continue
            if gp is None:
                continue
            gl = gp[0]
            for cy in cys:
                for t in ("line", 200, 230):
                    try:
                        gl.add_plot(wsd, coly=cy, colx=cx, type=t)
                        break
                    except Exception:
                        continue
            try:
                gl.rescale()
                gl.set_xlabel(xlab)
                gl.set_ylabel(ylab)
                gp.lname = _sanitizar_nombre(titulo)
            except Exception:
                pass
        except Exception as exc:
            print(f"     (gráfica '{titulo}' omitida: {exc})")


# ═════════════════════════════════════════════════════════════════════════════
# PROCESO
# ═════════════════════════════════════════════════════════════════════════════

def procesar(tst_file: str, id_ensayo: str, P_base: dict, opts: dict):
    bof = _mod_bof()
    P = {**P_base, "tst_file": tst_file, "id_ensayo": id_ensayo}

    print(f"\n{'='*60}\n  {id_ensayo}\n{'='*60}")

    df = bof.leer_tst(tst_file)
    df, stats, _df_p = bof.calcular(df, P)   # añade distancia_m, *_roll, Fx_abs, P['stroke_mm']
    df["tiempo_min"] = df["tiempo_s"] / 60.0

    out_dir = Path(tst_file).parent
    sep     = _delim_char(opts["delimitador"])
    ext     = ".csv" if opts["delimitador"] == "coma" else ".dat"
    saved   = []

    # Serie completa
    p_full = out_dir / f"{id_ensayo}_origin{ext}"
    escribir_serie(df, p_full, sep)
    saved.append(p_full)

    # Serie diezmada (opcional)
    decim = opts.get("decimado", 1)
    if decim > 1:
        p_dec = out_dir / f"{id_ensayo}_origin_dec{decim}{ext}"
        escribir_serie(df.iloc[::decim], p_dec, sep)
        saved.append(p_dec)

    print("  Guardado:")
    for f in saved:
        print(f"    {f}")

    return P, df, _resumen_dict(P, stats)


def main():
    dlg = OriginDialog()
    if dlg.cancelled:
        print("Operación cancelada.")
        return

    total  = len(dlg.files)
    sep    = _delim_char(dlg.opts["delimitador"])
    errors = []
    series = []          # [(P, df)]  para el .opju
    filas  = []          # [dict]     KPIs de esta ejecución

    print(f"\nExportando {total} ensayo(s) a OriginLab...\n")

    for i, (tst_file, id_ensayo) in enumerate(dlg.files, 1):
        print(f"[{i}/{total}]", end=" ")
        try:
            P, df, fila = procesar(tst_file, id_ensayo, dlg.params, dlg.opts)
            series.append((P, df))
            filas.append(fila)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append((id_ensayo, str(exc)))

    # ── Resumen KPIs acumulativo ──────────────────────────────────
    if filas:
        base_dir = Path(dlg.files[0][0]).parent
        summ = Path(dlg.opts["resumen_path"]) if dlg.opts["resumen_path"] else None
        try:
            p = consolidar_resumen(filas, base_dir, summ, sep)
            print(f"\n  Resumen KPIs → {p}")
            print(f"  Histórico    → {p.with_suffix('.json')}")
        except Exception as exc:
            print(f"\n  AVISO: no se pudo escribir el resumen: {exc}")

    # ── Proyecto Origin (.opju) — uno por ensayo, <id>.opju ───────
    if dlg.opts["opju"] and series:
        print(f"\n  Generando proyecto Origin por ensayo ({len(series)})...")
        exportar_opju(series, filas, dlg.opts["graficas"])

    print(f"\n{'='*60}")
    print(f"  Completados: {total - len(errors)}/{total}")
    if errors:
        print("  Errores:")
        for name, msg in errors:
            print(f"    {name}: {msg}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
