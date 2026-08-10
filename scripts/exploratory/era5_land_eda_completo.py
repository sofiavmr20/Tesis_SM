"""
===============================================================================
ANÁLISIS EXPLORATORIO DE DATOS (EDA) - ERA5-LAND MÉRIDA (2020-2026)
===============================================================================

Objetivo: Identificar zonas de clima homogéneo para modelar precipitación
orientada a pronóstico de crecidas repentinas.

Dataset: ERA5-Land horario, 19×17 píxeles (~9km), Mérida, Venezuela.
         Lat: 7.5°N – 9.3°N, Lon: -72.0°W – -70.4°W.
         Periodo: 2020-01-01 a 2026-03-31.

NOTA CRÍTICA: Las variables acumuladas de ERA5-Land (tp, e, sro, ssro,
ssrd, strd, sshf, slhf y componentes de evaporación) vienen como sumas
acumuladas dentro de cada ciclo de pronóstico (reset a las ~01 UTC /
21:00 UTC-4). Se aplica diff() intrapíxel para obtener valores horarios
correctos.

Estructura:
  PARTE 1 → Carga, diferenciación de variables acumuladas y estadísticas
  PARTE 2 → Análisis espacial de precipitación y variables clave
  PARTE 3 → Análisis temporal (ciclo diurno, estacionalidad, extremos)
  PARTE 4 → Clustering multivariado para zonas climáticas homogéneas
  PARTE 5 → Caracterización de zonas y síntesis
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
OUTPUT_FIG = Path("output/figures")
OUTPUT_REP = Path("output/reports")
OUTPUT_FIG.mkdir(parents=True, exist_ok=True)
OUTPUT_REP.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = Path("data/processed/era5_land_merida_2020_2026_utc4.parquet")

_ts_print: list[str] = []


def ts(msg: str = "", header: str = "") -> None:
    now = time.strftime("%H:%M:%S")
    line = f"[{now}] {header}{msg}" if header else f"[{now}] {msg}" if msg else ""
    if line:
        print(line)
    _ts_print.append(line)


def save_report(filename: str) -> None:
    (OUTPUT_REP / filename).write_text("\n".join(_ts_print), encoding="utf-8")

# Variables que requieren diferenciación (acumuladas en pronóstico)
ACCUM_VARS = ["tp", "e", "sro", "ssro", "ssrd", "strd", "sshf", "slhf",
              "evatc", "evabs", "evaow", "evavt"]
# Variables instantáneas (sin diferenciar)
INST_VARS = ["t2m", "d2m", "sp", "u10", "v10",
             "swvl1", "swvl2", "swvl3", "stl1", "lai_lv", "lai_hv"]


def differentiate_accumulated(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte variables acumuladas de ERA5-Land a valores horarios reales."""
    df = df.copy()
    df = df.sort_values(["latitude", "longitude", "time_utc4"])
    for var in ACCUM_VARS:
        if var in df.columns:
            df[f"{var}_raw"] = df[var]
            diffed = df.groupby(["latitude", "longitude"])[var].diff()
            diffed = diffed.fillna(df[var])
            mask_neg = diffed < 0
            diffed.loc[mask_neg] = df.loc[mask_neg, var]
            mask_still_neg = diffed < 0
            diffed.loc[mask_still_neg] = 0
            df[var] = diffed.astype("float32")
    return df


# ===========================================================================
# PARTE 1: CARGA Y ESTADÍSTICAS DESCRIPTIVAS
# ===========================================================================
def parte1_carga_exploracion() -> pd.DataFrame:
    ts("=" * 70)
    ts("PARTE 1: CARGA, DIFERENCIACIÓN Y ESTADÍSTICAS DESCRIPTIVAS")
    ts("=" * 70)

    t0 = time.time()
    df = pd.read_parquet(PARQUET_PATH)
    ts(f"Carga: {time.time() - t0:.1f} s  |  {df.shape[0]:,} filas × {df.shape[1]} columnas  |  "
       f"{df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

    lats = sorted(df["latitude"].unique())
    lons = sorted(df["longitude"].unique())
    ts(f"Rejilla: {len(lats)}×{len(lons)} = {len(lats) * len(lons)} píxeles")
    ts(f"  Lat: {lats[0]:.1f}–{lats[-1]:.1f}°N  (Δ={lats[1] - lats[0]:.1f}°)")
    ts(f"  Lon: {lons[0]:.2f}–{lons[-1]:.2f}°W  (Δ={lons[1] - lons[0]:.1f}°)")
    ts(f"  Periodo: {df['time_utc4'].min().date()} → {df['time_utc4'].max().date()}")

    # Nulos (píxeles del Lago de Maracaibo)
    nulls = df.isnull().sum()
    nulls_sig = nulls[nulls > 0]
    if not nulls_sig.empty:
        ts(f"\nValores nulos: {int(nulls_sig.iloc[0]):,} filas por variable ({int(nulls_sig.iloc[0] / df.groupby(['latitude', 'longitude']).size().max())} píxeles lacustres)")
        ts("  Variables afectadas: " + ", ".join(nulls_sig.index.tolist()))

    # Diferenciar variables acumuladas
    ts("\nAplicando diff() a variables acumuladas de ERA5-Land...")
    t1 = time.time()
    df = differentiate_accumulated(df)
    ts(f"  Variables transformadas: {ACCUM_VARS}")
    ts(f"  Tiempo: {time.time() - t1:.1f} s")

    # Estadísticas descriptivas con valores horarios corregidos
    ts("\nEstadísticas descriptivas (valores horarios corregidos):")
    desc_vars = ["tp", "e", "sro", "t2m", "swvl1", "sp"]
    desc_labels = {
        "tp": "Precipitación (mm/h)",
        "e": "Evaporación (mm/h)",
        "sro": "Escorrentía sup. (mm/h)",
        "t2m": "Temperatura 2m (°C)",
        "swvl1": "Hum. suelo 0-7cm (m³/m³)",
        "sp": "Presión sup. (hPa)",
    }
    desc_factors = {"tp": 1000, "e": 1000, "sro": 1000, "t2m": -273.15, "swvl1": 1, "sp": 0.01}
    desc_units = {"tp": "mm/h", "e": "mm/h", "sro": "mm/h", "t2m": "°C", "swvl1": "m³/m³", "sp": "hPa"}

    for var in desc_vars:
        vals = df[var].dropna() + desc_factors[var] if var == "t2m" else df[var].dropna() * desc_factors[var]
        q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
        ts(f"  {desc_labels[var]:35s} μ={vals.mean():.3f}  σ={vals.std():.3f}  "
           f"Q50={q50:.3f}  [{q25:.3f}, {q75:.3f}]")

    save_report("01_descripcion.txt")
    return df


# ===========================================================================
# PARTE 2: ANÁLISIS ESPACIAL
# ===========================================================================
def parte2_espacial(df: pd.DataFrame) -> pd.DataFrame:
    ts("=" * 70)
    ts("PARTE 2: ANÁLISIS ESPACIAL DE PRECIPITACIÓN Y VARIABLES CLAVE")
    ts("=" * 70)

    lats = sorted(df["latitude"].unique(), reverse=True)
    lons = sorted(df["longitude"].unique())
    extent = [min(lons), max(lons), min(lats), max(lats)]

    def array_a_grid(series):
        grid = np.full((len(lats), len(lons)), np.nan)
        lat_map = {lat: i for i, lat in enumerate(lats)}
        lon_map = {lon: i for i, lon in enumerate(lons)}
        for (lat, lon), val in series.items():
            grid[lat_map[lat], lon_map[lon]] = val
        return grid

    # Agregar columnas temporales con unidades prácticas
    df_w = df.dropna(subset=["tp"]).copy()
    df_w["year"] = df_w["time_utc4"].dt.year
    df_w["month"] = df_w["time_utc4"].dt.month
    df_w["hour"] = df_w["time_utc4"].dt.hour
    df_w["tp_mmh"] = df_w["tp"] * 1000
    df_w["e_mmh"] = df_w["e"] * 1000
    df_w["sro_mmh"] = df_w["sro"] * 1000
    df_w["t2m_c"] = df_w["t2m"] - 273.15

    # --- Mapa 1: Precipitación anual media por píxel ---
    ts("Mapeando precipitación anual media por píxel...")
    n_years = df_w["year"].nunique()
    yearly_by_pixel = df_w.groupby(["latitude", "longitude", "year"])["tp_mmh"].sum()
    annual_mean = yearly_by_pixel.groupby(["latitude", "longitude"]).mean()

    tp_annual_grid = array_a_grid(annual_mean.clip(upper=10000))
    ts(f"  Rango: {annual_mean.min():.0f} – {annual_mean.max():.0f} mm/año  "
       f"(μ={annual_mean.mean():.0f}, Q50={annual_mean.median():.0f})")

    # --- Mapa 2: Intensidad media de lluvia (>0 mm/h) ---
    ts("Mapeando intensidad media de lluvia (>0.1 mm/h)...")
    df_rain = df_w[df_w["tp_mmh"] > 0.1]
    intensity = df_rain.groupby(["latitude", "longitude"])["tp_mmh"].mean()
    intensity_grid = array_a_grid(intensity.clip(upper=10))
    ts(f"  Rango: {intensity.min():.2f} – {intensity.max():.2f} mm/h")

    # --- Mapa 3: Temperatura media ---
    ts("Mapeando temperatura media...")
    temp_mean = df_w.groupby(["latitude", "longitude"])["t2m_c"].mean()
    temp_grid = array_a_grid(temp_mean)
    ts(f"  Rango: {temp_mean.min():.1f} – {temp_mean.max():.1f} °C")

    # --- Mapa 4: Humedad de suelo ---
    ts("Mapeando humedad de suelo...")
    swvl = df_w.groupby(["latitude", "longitude"])["swvl1"].mean()
    swvl_grid = array_a_grid(swvl)
    ts(f"  Rango: {swvl.min():.3f} – {swvl.max():.3f} m³/m³")

    # --- Mapa 5: Escorrentía anual ---
    sro_annual = df_w.groupby(["latitude", "longitude", "year"])["sro_mmh"].sum()
    sro_annual_mean = sro_annual.groupby(["latitude", "longitude"]).mean()
    sro_grid = array_a_grid(sro_annual_mean.clip(upper=sro_annual_mean.quantile(0.95)))
    ts(f"  Escorrentía anual: {sro_annual_mean.median():.0f} mm/año (mediana)")

    # --- Mapa 6: Frecuencia de lluvia (>0.1 mm/h) ---
    rain_freq = (df_w["tp_mmh"] > 0.1).groupby([df_w["latitude"], df_w["longitude"]]).mean() * 100
    freq_grid = array_a_grid(rain_freq)

    # --- Figura compuesta ---
    ts("\nGenerando mapas espaciales...")
    map_configs = [
        (tp_annual_grid, "Precipitación Anual (mm/año)", "Blues", None),
        (intensity_grid, "Intensidad de Lluvia >0.1 mm/h (mm/h)", "YlOrRd", None),
        (temp_grid, "Temperatura 2m (°C)", "RdBu_r", temp_grid),
        (swvl_grid, "Humedad de Suelo 0-7 cm (m³/m³)", "YlGnBu", None),
        (sro_grid, "Escorrentía Superficial Anual (mm/año)", "Purples", None),
        (freq_grid, "Frecuencia de Lluvia >0.1 mm/h (%)", "Greens", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (grid, title, cmap, special_norm) in zip(axes.flat, map_configs):
        if special_norm is not None and id(special_norm) == id(temp_grid):
            vc = np.nanmedian(grid)
            vm = max(np.nanmax(abs(grid - vc)), 0.01)
            norm = mcolors.TwoSlopeNorm(vmin=vc - vm, vcenter=vc, vmax=vc + vm)
        else:
            norm = None
        im = ax.imshow(grid, cmap=cmap, aspect="auto", extent=extent, origin="upper", norm=norm)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitud (°W)")
        ax.set_ylabel("Latitud (°N)")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Patrones Espaciales – ERA5-Land Mérida (2020–2026)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIG / "02_mapas_espaciales.png")
    plt.close()
    ts("  → output/figures/02_mapas_espaciales.png")

    # --- Gradiente orográfico (latitud como proxy de elevación) ---
    ts("Analizando gradiente latitudinal de precipitación...")
    ny = df_w["year"].nunique()
    tp_lat = df_w.groupby(["latitude", "month"])["tp_mmh"].sum().unstack().fillna(0)
    tp_lat_annual = tp_lat.sum(axis=1) / ny
    month_names_s = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Perfil
    axes[0].plot(tp_lat.index, tp_lat_annual.values, "o-", color="#2c3e50", lw=2, markersize=5)
    axes[0].fill_between(tp_lat.index, 0, tp_lat_annual.values, color="#3498db", alpha=0.3)
    axes[0].set_xlabel("Latitud (°N)")
    axes[0].set_ylabel("Precipitación anual (mm)")
    axes[0].set_title("Perfil Latitudinal de Precipitación Anual", fontweight="bold")
    axes[0].invert_xaxis()

    # Barras horizontales
    axes[1].barh(tp_lat.index, tp_lat_annual.values, color="#2ecc71", edgecolor="white")
    axes[1].set_xlabel("Precipitación anual (mm)")
    axes[1].set_ylabel("Latitud (°N)")
    axes[1].set_title("Precipitación por Latitud", fontweight="bold")
    axes[1].invert_yaxis()

    # Heatmap estacional por latitud
    im = axes[2].imshow(tp_lat.values, aspect="auto", cmap="Blues")
    axes[2].set_xticks(range(12))
    axes[2].set_xticklabels(month_names_s)
    axes[2].set_yticks(range(len(tp_lat.index)))
    axes[2].set_yticklabels([f"{lat:.1f}°N" for lat in tp_lat.index])
    axes[2].set_xlabel("Mes")
    axes[2].set_ylabel("Latitud")
    axes[2].set_title("Ciclo Estacional × Latitud (mm/mes)", fontweight="bold")
    plt.colorbar(im, ax=axes[2], label="mm/mes", shrink=0.8)

    fig.suptitle("Variabilidad Espacial de la Precipitación", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIG / "02_gradiente_precipitacion.png")
    plt.close()
    ts("  → output/figures/02_gradiente_precipitacion.png")

    save_report("02_espacial.txt")
    return df_w


# ===========================================================================
# PARTE 3: ANÁLISIS TEMPORAL
# ===========================================================================
def parte3_temporal(df_w: pd.DataFrame) -> pd.DataFrame:
    ts("=" * 70)
    ts("PARTE 3: ANÁLISIS TEMPORAL DE PRECIPITACIÓN")
    ts("=" * 70)

    # --- 3A. Ciclo diurno ---
    ts("Ciclo diurno de precipitación:")
    diurno = df_w.groupby("hour")["tp_mmh"].mean()
    for h, val in diurno.nlargest(3).items():
        ts(f"  {int(h):02d}:00 VET → {val:.4f} mm/h (promedio)")

    # --- 3B. Ciclo anual (estacionalidad) ---
    ts("\nCiclo anual:")
    anual = df_w.groupby("month")["tp_mmh"].sum() / df_w["year"].nunique()
    for m, val in anual.items():
        ts(f"  Mes {int(m):2d}: {val:.1f} mm/mes")
    ts(f"  Máx: Mes {int(anual.idxmax())} ({anual.max():.0f} mm)")
    ts(f"  Mín: Mes {int(anual.idxmin())} ({anual.min():.0f} mm)")

    # --- 3C. Variabilidad interanual ---
    ts("\nVariabilidad interanual:")
    interanual = df_w.groupby("year")["tp_mmh"].sum().div(df_w.groupby("year")[["latitude", "longitude"]].nunique().sum(axis=1) / 2)
    ts(f"  Media: {interanual.mean():.0f} mm/año/píxel")
    ts(f"  CV:    {interanual.std() / interanual.mean() * 100:.1f}%")
    for yr, val in interanual.items():
        ts(f"  {int(yr)}: {val:.0f} mm")

    # --- 3D. Eventos extremos ---
    ts("\nEventos extremos de precipitación horaria:")
    tp_v = df_w["tp_mmh"]
    q90, q95, q99, q999 = tp_v.quantile([0.90, 0.95, 0.99, 0.999])
    ts(f"  P90  = {q90:.1f} mm/h")
    ts(f"  P95  = {q95:.1f} mm/h")
    ts(f"  P99  = {q99:.2f} mm/h")
    ts(f"  P99.9 = {q999:.2f} mm/h")
    ts(f"  Máximo registrado: {tp_v.max():.2f} mm/h")

    # Rachas de lluvia intensa
    df_sorted = df_w.sort_values(["latitude", "longitude", "time_utc4"])
    df_sorted["heavy"] = df_sorted["tp_mmh"] > q95
    shift_ref = df_sorted.groupby(["latitude", "longitude"])["heavy"].shift().fillna(False)
    df_sorted["grupo"] = (
        (df_sorted["heavy"] != shift_ref)
        .groupby([df_sorted["latitude"], df_sorted["longitude"]])
        .cumsum()
    )
    rachas = df_sorted[df_sorted["heavy"]].groupby(["latitude", "longitude", "grupo"]).size()
    if len(rachas) > 0:
        ts(f"\n  Rachas de lluvia intensa (>P95):")
        ts(f"    Duración media: {rachas.mean():.1f} h")
        ts(f"    Duración máxima: {rachas.max()} h")
        ts(f"    Total de rachas: {len(rachas):,}")
        ts(f"    Máximo de precipitación en una racha: {df_sorted['tp_mmh'][df_sorted['heavy']].max():.1f} mm/h")

    # --- 3E. Figura de análisis temporal ---
    ts("\nGenerando figura de análisis temporal...")
    month_names_f = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Ciclo diurno
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(diurno.index, diurno.values, color="#3498db", alpha=0.6)
    ax1.plot(diurno.index, diurno.values, "o-", color="#2c3e50", lw=1.5, markersize=4)
    ax1.set_xlabel("Hora Local (VET)")
    ax1.set_ylabel("Precipitación (mm/h)")
    ax1.set_title("Ciclo Diurno", fontweight="bold")
    ax1.set_xticks(range(0, 24, 3))

    # Ciclo anual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(anual.index, anual.values, color="#2ecc71", edgecolor="white")
    ax2.set_xlabel("Mes")
    ax2.set_ylabel("Precipitación (mm/mes)")
    ax2.set_title("Ciclo Anual", fontweight="bold")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names_f)

    # Interanual
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(interanual.index.astype(int), interanual.values, color="#e74c3c", edgecolor="white")
    ax3.axhline(interanual.mean(), color="black", ls="--", lw=1, label=f"μ={interanual.mean():.0f} mm")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("Precipitación Total (mm)")
    ax3.set_title("Variabilidad Interanual", fontweight="bold")
    ax3.legend()

    # Distribución de precipitación horaria
    ax4 = fig.add_subplot(gs[1, 0])
    tp_pos = tp_v[tp_v > 0.01]
    ax4.hist(np.log10(tp_pos.clip(lower=0.01)), bins=60, color="#9b59b6", alpha=0.7,
             edgecolor="white", density=True)
    ax4.set_xlabel("log₁₀(Precipitación mm/h)")
    ax4.set_ylabel("Densidad")
    ax4.set_title("Distribución de Lluvia Horaria (>0.01 mm/h)", fontweight="bold")

    # Eventos extremos por mes
    ax5 = fig.add_subplot(gs[1, 1])
    extreme_m = df_sorted.groupby(df_sorted["time_utc4"].dt.month)["heavy"].sum()
    ax5.bar(extreme_m.index, extreme_m.values, color="#e67e22", edgecolor="white")
    ax5.set_xlabel("Mes")
    ax5.set_ylabel("Horas > P95")
    ax5.set_title("Eventos Extremos por Mes", fontweight="bold")
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(month_names_f)

    # Serie temporal diaria
    ax6 = fig.add_subplot(gs[1, 2])
    daily = df_w.groupby(df_w["time_utc4"].dt.date)["tp_mmh"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily_smooth = daily.rolling(30, center=True).mean()
    ax6.plot(daily.index, daily.values, color="#3498db", alpha=0.15, lw=0.5)
    ax6.plot(daily_smooth.index, daily_smooth.values, color="#2c3e50", lw=1.5)
    ax6.set_xlabel("Fecha")
    ax6.set_ylabel("Precipitación (mm/día)")
    ax6.set_title("Serie Temporal de Precipitación Diaria\n(promedio espacial, suavizado 30d)", fontweight="bold")

    fig.suptitle("Análisis Temporal de la Precipitación – Mérida, Venezuela", fontweight="bold", fontsize=15)
    fig.savefig(OUTPUT_FIG / "03_analisis_temporal.png")
    plt.close()
    ts("  → output/figures/03_analisis_temporal.png")

    save_report("03_temporal.txt")
    return df_w


# ===========================================================================
# PARTE 4: CLUSTERING PARA ZONAS CLIMÁTICAS HOMOGÉNEAS
# ===========================================================================
def parte4_clustering(df_w: pd.DataFrame) -> dict:
    ts("=" * 70)
    ts("PARTE 4: IDENTIFICACIÓN DE ZONAS CLIMÁTICAS HOMOGÉNEAS (CLUSTERING)")
    ts("=" * 70)

    unique_lats = sorted(df_w["latitude"].unique(), reverse=True)
    unique_lons = sorted(df_w["longitude"].unique())
    extent = [min(unique_lons), max(unique_lons), min(unique_lats), max(unique_lats)]

    def array_a_grid(series):
        grid = np.full((len(unique_lats), len(unique_lons)), np.nan)
        lat_m = {lat: i for i, lat in enumerate(unique_lats)}
        lon_m = {lon: i for i, lon in enumerate(unique_lons)}
        for (lat, lon), val in series.items():
            grid[lat_m[lat], lon_m[lon]] = val
        return grid

    # --- 4A. Matriz de características por píxel ---
    ts("Construyendo características climáticas por píxel...")

    feat_base = ["tp_mmh", "t2m_c", "swvl1", "e_mmh", "sp", "sro_mmh", "u10", "v10", "d2m"]

    features = []
    for (lat, lon), grp in df_w.groupby(["latitude", "longitude"]):
        f = {"latitude": lat, "longitude": lon}
        for var in feat_base:
            vals = grp[var]
            f[f"{var}_mean"] = vals.mean()
            f[f"{var}_std"] = vals.std()
            f[f"{var}_p10"] = vals.quantile(0.10)
            f[f"{var}_p90"] = vals.quantile(0.90)

        # Precipitación anual media
        yearly_sums = grp.groupby("year")["tp_mmh"].sum()
        f["tp_anual_mm"] = yearly_sums.mean()

        # Intensidad media de lluvia (>0.1 mm/h)
        tp_pos = grp["tp_mmh"][grp["tp_mmh"] > 0.1]
        f["tp_intensidad"] = tp_pos.mean() if len(tp_pos) > 0 else 0

        # Frecuencia de lluvia
        f["tp_freq"] = (grp["tp_mmh"] > 0.1).mean()

        # Frecuencia de eventos extremos (>P95 global)
        f["tp_extremos"] = (grp["tp_mmh"] > grp["tp_mmh"].quantile(0.95)).mean()

        # Estacionalidad
        grp_s = grp.copy()
        grp_s["season"] = grp_s["month"].apply(
            lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5) else "JJA" if m in (6, 7, 8) else "SON"
        )
        tp_season = grp_s.groupby("season")["tp_mmh"].sum()
        f["tp_ratio_jja"] = tp_season.get("JJA", 0) / max(tp_season.sum(), 1)
        f["tp_ratio_djf"] = tp_season.get("DJF", 0) / max(tp_season.sum(), 1)
        f["tp_ratio_mam"] = tp_season.get("MAM", 0) / max(tp_season.sum(), 1)
        f["tp_ratio_son"] = tp_season.get("SON", 0) / max(tp_season.sum(), 1)

        # Escorrentía anual
        sro_sum = grp.groupby("year")["sro_mmh"].sum().mean()
        f["sro_anual_mm"] = sro_sum

        features.append(f)

    df_feat = pd.DataFrame(features).set_index(["latitude", "longitude"])
    ts(f"  Dimensiones: {df_feat.shape[0]} píxeles × {df_feat.shape[1]} variables")

    # Limpiar columnas con varianza nula o NaNs
    df_clean = df_feat.dropna(axis=1).copy()
    low_var = df_clean.columns[df_clean.std() < 1e-10]
    if len(low_var) > 0:
        df_clean = df_clean.drop(columns=low_var)
        ts(f"  Columnas eliminadas (var≈0): {list(low_var)}")
    ts(f"  Matriz final: {df_clean.shape}")

    # --- 4B. PCA ---
    ts("\nAplicando PCA...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean.values)

    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    nc95 = int(np.searchsorted(cumvar, 0.95) + 1)
    ts(f"  Componentes para 95% varianza: {nc95} de {len(cumvar)}")

    pca = PCA(n_components=nc95)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    for i in range(min(5, nc95)):
        ts(f"  PC{i+1}: {var_exp[i]:.1f}%  (Σ={var_exp[:i+1].sum():.1f}%)")

    # Loadings
    ts("\n  Loadings principales (|loading|>0.1):")
    for pi in range(nc95):
        loads = pca.components_[pi]
        top = np.argsort(np.abs(loads))[::-1][:3]
        ts(f"    PC{pi+1} ({var_exp[pi]:.1f}%): {', '.join(f'{df_clean.columns[i]}={loads[i]:+.2f}' for i in top)}")

    # --- 4C. K óptimo ---
    ts("\nBúsqueda de K óptimo...")
    K_range = range(2, min(9, X_pca.shape[0]))
    inertias, sil_scores = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labs = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labs))
    k_opt = K_range[np.argmax(sil_scores)]
    ts(f"  K óptimo = {k_opt} (silhouette = {max(sil_scores):.4f})")

    # --- 4D. Clustering final ---
    ts("\nClustering final K-Means...")
    km = KMeans(n_clusters=k_opt, random_state=42, n_init=30)
    labels = km.fit_predict(X_pca)
    sil_fin = silhouette_score(X_pca, labels)
    ts(f"  Silhouette: {sil_fin:.4f}")

    df_feat["cluster"] = labels
    for c, n in df_feat["cluster"].value_counts().sort_index().items():
        ts(f"  Zona {c}: {n} píxeles ({n / len(df_feat) * 100:.1f}%)")

    # Validación jerárquica
    Z = linkage(X_pca, method="ward")
    labels_hc = fcluster(Z, t=k_opt, criterion="maxclust") - 1
    ari = adjusted_rand_score(labels, labels_hc)
    ts(f"  Índice Rand Ajustado (KMeans vs Jerárquico): {ari:.4f}")

    # --- 4E. Figuras de clustering ---
    ts("\nGenerando figuras de clustering...")
    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # PCA variance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(cumvar) + 1), cumvar * 100, "o-", color="#2c3e50", lw=1.5, markersize=3)
    ax1.axhline(95, color="#e74c3c", ls="--", alpha=0.7)
    ax1.axvline(nc95, color="#e74c3c", ls="--", alpha=0.7)
    ax1.set_xlabel("N° Componentes")
    ax1.set_ylabel("Varianza Acumulada (%)")
    ax1.set_title("Varianza PCA", fontweight="bold")
    ax1.set_ylim(0, 105)

    # Elbow + silhouette
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_s = ax2.twinx()
    ax2.plot(K_range, inertias, "o-", color="#3498db", lw=2, label="Inercia")
    ax2_s.plot(K_range, sil_scores, "s--", color="#e74c3c", lw=2, label="Silhouette")
    ax2.axvline(k_opt, color="green", ls="--", alpha=0.7)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Inercia", color="#3498db")
    ax2_s.set_ylabel("Silhouette", color="#e74c3c")
    ax2.set_title("Selección de K", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2_s.legend(loc="upper right")

    # Dendrograma
    ax3 = fig.add_subplot(gs[0, 2])
    dendrogram(Z, ax=ax3, truncate_mode="level", p=5, leaf_font_size=8,
               color_threshold=0.7 * Z[-k_opt + 1, 2])
    ax3.set_title("Dendrograma (Ward)", fontweight="bold")
    ax3.set_xlabel("Índice de Píxel")

    # Mapa de zonas
    ax4 = fig.add_subplot(gs[1, 0])
    grid_c = np.full((len(unique_lats), len(unique_lons)), np.nan)
    for (lat, lon), row in df_feat.iterrows():
        li = list(unique_lats).index(lat)
        lo = list(unique_lons).index(lon)
        grid_c[li, lo] = row["cluster"]
    im4 = ax4.imshow(grid_c, cmap=cmap, aspect="auto", extent=extent, origin="upper", interpolation="nearest")
    ax4.set_title(f"Zonas Climáticas Homogéneas (K={k_opt})", fontweight="bold")
    ax4.set_xlabel("Longitud (°W)")
    ax4.set_ylabel("Latitud (°N)")

    # PC1 vs PC2
    ax5 = fig.add_subplot(gs[1, 1])
    for c in range(k_opt):
        m = labels == c
        ax5.scatter(X_pca[m, 0], X_pca[m, 1], c=[cmap(c)], label=f"Zona {c}",
                    s=50, edgecolors="white", linewidth=0.5, alpha=0.85)
    ax5.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax5.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax5.set_title("Proyección PCA", fontweight="bold")
    ax5.legend(fontsize=7, loc="best")

    # Precipitación anual por zona
    ax6 = fig.add_subplot(gs[1, 2])
    tp_z = df_feat.groupby("cluster")["tp_anual_mm"].mean().sort_values()
    ax6.bar(range(len(tp_z)), tp_z.values, color=[cmap(c) for c in tp_z.index], edgecolor="white")
    ax6.set_xticks(range(len(tp_z)))
    ax6.set_xticklabels([f"Z{c}" for c in tp_z.index])
    ax6.set_ylabel("mm/año")
    ax6.set_title("Precipitación Anual Promedio por Zona", fontweight="bold")

    fig.suptitle("Zonas Climáticas Homogéneas – Mérida, Venezuela", fontweight="bold", fontsize=15)
    fig.savefig(OUTPUT_FIG / "04_clustering_zonas.png")
    plt.close()
    ts("  → output/figures/04_clustering_zonas.png")

    # Silhouette detallado
    fig2, ax_sil = plt.subplots(figsize=(10, 3))
    sil_vals = silhouette_samples(X_pca, labels)
    y_low = 10
    for c in range(k_opt):
        cv = sil_vals[labels == c]
        cv.sort()
        ax_sil.fill_betweenx(np.arange(y_low, y_low + len(cv)), cv, alpha=0.7, color=cmap(c), label=f"Zona {c}")
        y_low += len(cv) + 10
    ax_sil.axvline(sil_fin, color="red", ls="--", lw=1.5)
    ax_sil.set_xlabel("Coeficiente de Silhouette")
    ax_sil.set_ylabel("Píxeles")
    ax_sil.set_title("Silhouette por Píxel y Zona", fontweight="bold")
    ax_sil.legend(fontsize=8, ncol=k_opt)
    fig2.savefig(OUTPUT_FIG / "04c_silhouette_por_zona.png")
    plt.close(fig2)
    ts("  → output/figures/04c_silhouette_por_zona.png")

    df_feat.to_parquet(OUTPUT_REP / "04_cluster_features.parquet")
    ts("  → output/reports/04_cluster_features.parquet")
    save_report("04_clustering.txt")

    return {"features": df_feat, "labels": labels, "k_opt": k_opt, "silhouette": sil_fin,
            "X_pca": X_pca, "pca": pca, "var_exp": var_exp}


# ===========================================================================
# PARTE 5: CARACTERIZACIÓN DE ZONAS Y SÍNTESIS
# ===========================================================================
def parte5_caracterizacion(df_w: pd.DataFrame, clr: dict) -> None:
    ts("=" * 70)
    ts("PARTE 5: CARACTERIZACIÓN DE ZONAS CLIMÁTICAS Y SÍNTESIS")
    ts("=" * 70)

    df_feat = clr["features"]
    k_opt = clr["k_opt"]
    coord2cluster = df_feat["cluster"].to_dict()

    # Etiquetar datos horarios con zona climática
    df_w["cluster"] = df_w.apply(lambda r: coord2cluster.get((r["latitude"], r["longitude"]), -1), axis=1)
    df_labeled = df_w[df_w["cluster"] >= 0]
    n_years = df_labeled["year"].nunique()

    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    season_order = ["DJF", "MAM", "JJA", "SON"]

    # --- 5A. Perfil detallado por zona ---
    for c in range(k_opt):
        n_pix = int(df_feat["cluster"].value_counts().get(c, 0))
        cdata = df_labeled[df_labeled["cluster"] == c]

        tp_year = cdata.groupby("year")["tp_mmh"].sum().mean() / n_pix
        t2m_avg = cdata["t2m_c"].mean()
        tp_int = cdata["tp_mmh"][cdata["tp_mmh"] > 0.1].mean()
        tp_frq = (cdata["tp_mmh"] > 0.1).mean() * 100
        sro_yr = cdata.groupby("year")["sro_mmh"].sum().mean() / n_pix

        coords = df_feat[df_feat["cluster"] == c].index.tolist()
        lats_p = [cc[0] for cc in coords]

        ts(f"\n{'─'*40}")
        ts(f"  ZONA CLIMÁTICA {c}  ({n_pix} píxeles)")
        ts(f"{'─'*40}")
        ts(f"  Precipitación anual:      {tp_year:.0f} mm/año")
        ts(f"  Intensidad lluvia (>0.1): {tp_int:.2f} mm/h")
        ts(f"  Frecuencia de lluvia:     {tp_frq:.1f}%")
        ts(f"  Temperatura media 2m:     {t2m_avg:.1f} °C")
        ts(f"  Escorrentía superficial:  {sro_yr:.0f} mm/año")
        ts(f"  Latitud:                  {min(lats_p):.1f}°N – {max(lats_p):.1f}°N")

        # Estacionalidad
        ts("  Régimen estacional:")
        cdata["season"] = cdata["month"].map(season_map)
        for s in season_order:
            sd = cdata[cdata["season"] == s]
            tp_s = sd["tp_mmh"].sum() / n_pix / n_years
            ts(f"    {s}: {tp_s:.0f} mm/trimestre")

    # --- 5B. Figura: Ciclos por zona ---
    ts("\n\nGenerando figura: Ciclos de precipitación por zona climática...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    cmap = plt.cm.tab10

    # Diurno por zona
    diurno_z = df_labeled.groupby(["cluster", "hour"])["tp_mmh"].mean().unstack(level=0)
    for c in range(k_opt):
        axes[0].plot(diurno_z.index, diurno_z[c], "o-", label=f"Zona {c}",
                     lw=1.5, markersize=3, alpha=0.85, color=cmap(c))
    axes[0].set_xlabel("Hora Local (VET)")
    axes[0].set_ylabel("Precipitación (mm/h)")
    axes[0].set_title("Ciclo Diurno de Precipitación por Zona", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(range(0, 24, 3))

    # Anual por zona
    anual_z = df_labeled.groupby(["cluster", "month"])["tp_mmh"].sum().unstack(level=0) / n_years
    meses_full = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    for c in range(k_opt):
        axes[1].plot(range(1, 13), anual_z[c], "o-", label=f"Zona {c}",
                     lw=1.5, markersize=4, alpha=0.85, color=cmap(c))
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(meses_full, rotation=45)
    axes[1].set_xlabel("Mes")
    axes[1].set_ylabel("Precipitación (mm/mes)")
    axes[1].set_title("Ciclo Anual de Precipitación por Zona", fontweight="bold")
    axes[1].legend(fontsize=8)

    fig.suptitle("Regímenes de Precipitación por Zona Climática Homogénea", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIG / "05_ciclos_por_zona.png")
    plt.close()
    ts("  → output/figures/05_ciclos_por_zona.png")

    # --- 5C. Síntesis final ---
    ts("\n" + "=" * 70)
    ts("SÍNTESIS DEL ANÁLISIS EXPLORATORIO")
    ts("=" * 70)
    ts(f"Zonas climáticas homogéneas identificadas: {k_opt}")
    ts(f"Índice de Silhouette:                   {clr['silhouette']:.4f}")
    ts("")
    ts("Variables clave por zona (para modelado de precipitación):")
    ts("  · Precipitación total anual, intensidad media, frecuencia")
    ts("  · Temperatura 2m y punto de rocío")
    ts("  · Humedad de suelo (swvl1–3)")
    ts("  · Evaporación y evapotranspiración")
    ts("  · Escorrentía superficial y sub-superficial")
    ts("  · Presión superficial (proxy de elevación)")
    ts("  · Radiación solar descendente")
    ts("")
    ts("Recomendaciones para pronóstico de crecidas repentinas:")
    ts("  1. Modelar precipitación de forma independiente por zona climática")
    ts("  2. Usar humedad de suelo antecedente como predictor de saturación")
    ts("  3. Incorporar escorrentía (sro, ssro) para calibración hidrológica")
    ts("  4. Considerar el ciclo diurno marcado (horas pico de lluvia)")
    ts("  5. Atender la estacionalidad (régimen bimodal/unimodal por zona)")
    ts("  6. Definir umbrales de alerta basados en percentiles por zona")
    ts("  7. La latitud funciona como buen proxy de gradiente orográfico")

    save_report("05_sintesis.txt")

    with open(OUTPUT_REP / "EDA_completo.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_ts_print))
    ts("\n  → Reporte completo: output/reports/EDA_completo.txt")


# ===========================================================================
# EJECUCIÓN PRINCIPAL
# ===========================================================================
def main():
    t0 = time.time()

    # PARTE 1: Carga + diferenciación + estadísticas
    df = parte1_carga_exploracion()

    # PARTE 2: Análisis espacial
    df_w = parte2_espacial(df)

    # PARTE 3: Análisis temporal
    df_w = parte3_temporal(df_w)

    # PARTE 4: Clustering
    clr = parte4_clustering(df_w)

    # PARTE 5: Caracterización
    parte5_caracterizacion(df_w, clr)

    elapsed = time.time() - t0
    ts(f"\n{'=' * 70}")
    ts(f"EDA COMPLETADO en {elapsed:.0f} s ({elapsed / 60:.1f} min)")
    ts(f"Resultados en output/figures/ y output/reports/")
    ts(f"{'=' * 70}")


if __name__ == "__main__":
    main()
