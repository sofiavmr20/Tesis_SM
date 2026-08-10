from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configuración estética de Matplotlib
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 14


def ejecutar_bloque3_eda():
    path_horario = Path("data/processed/era5_land_merida_2020_2026_utc4.parquet")
    path_diario = Path("data/processed/era5_land_merida_diario_2020_2026.parquet")

    if not path_diario.exists():
        print(f"Error: No se encuentra el dataset diario '{path_diario}'")
        return

    print("==================================================================")
    print(" BLOQUE 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA) - ERA5-LAND MÉRIDA")
    print("==================================================================")

    # -------------------------------------------------------------------
    # PARTE 1: PERFILES TEMPORALES Y CICLOS (DIURNO Y ANUAL)
    # -------------------------------------------------------------------
    print("\n--- 1. PROCESANDO PERFILES TEMPORALES Y CICLOS CLIMÁTICOS ---")
    t0 = time.time()

    # Leemos columnas necesarias del dataset horario para optimizar memoria
    cols_horarias = ["time_utc4", "t2m", "u10", "v10", "tp", "swvl1", "ssrd"]
    df_h = pd.read_parquet(path_horario, columns=cols_horarias)

    # Convertir unidades
    df_h["t2m_c"] = df_h["t2m"] - 273.15  # K a °C
    df_h["wind_speed"] = np.sqrt(df_h["u10"] ** 2 + df_h["v10"] ** 2)  # m/s
    df_h["tp_mm"] = df_h["tp"] * 1000.0  # m a mm/h

    # Derivar hora local y mes
    df_h["hour"] = df_h["time_utc4"].dt.hour
    df_h["month"] = df_h["time_utc4"].dt.month

    # 1A. Ciclo Diurno Promedio (0 a 23h VET)
    diurno_t2m = df_h.groupby("hour")["t2m_c"].mean()
    diurno_wind = df_h.groupby("hour")["wind_speed"].mean()
    diurno_tp = df_h.groupby("hour")["tp_mm"].mean()

    # 1B. Ciclo Anual Promedio (Enero a Diciembre)
    anual_t2m = df_h.groupby("month")["t2m_c"].mean()
    anual_tp = (
        df_h.groupby("month")["tp_mm"].sum() / df_h["time_utc4"].dt.year.nunique()
    )  # Acumulado promedio mensual
    anual_swvl = df_h.groupby("month")["swvl1"].mean()
    anual_ssrd = df_h.groupby("month")["ssrd"].mean() / 3600.0  # J/m2 a W/m2

    del df_h  # Liberar memoria del dataset horario

    # Graficar Perfiles Temporales (Ciclo Diurno y Anual)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Ciclo Diurno de Temperatura y Viento
    ax1 = axes[0, 0]
    ax1_wind = ax1.twinx()
    l1 = ax1.plot(
        diurno_t2m.index,
        diurno_t2m.values,
        color="#e74c3c",
        lw=2.5,
        label="Temperatura 2m (°C)",
    )
    l2 = ax1_wind.plot(
        diurno_wind.index,
        diurno_wind.values,
        color="#2980b9",
        lw=2.5,
        linestyle="--",
        label="Velocidad Viento (m/s)",
    )
    ax1.set_title(
        "Ciclo Diurno Promedio: Temperatura y Viento (UTC-4 VET)", fontweight="bold"
    )
    ax1.set_xlabel("Hora Local del Día (VET)")
    ax1.set_ylabel("Temperatura (°C)", color="#e74c3c")
    ax1_wind.set_ylabel("Velocidad del Viento (m/s)", color="#2980b9")
    ax1.set_xticks(range(0, 24, 2))

    # Panel 2: Ciclo Diurno de Precipitación por Hora
    ax2 = axes[0, 1]
    ax2.bar(diurno_tp.index, diurno_tp.values, color="#3498db", alpha=0.85, width=0.7)
    ax2.set_title("Ciclo Diurno Promedio: Precipitación (mm/hora)", fontweight="bold")
    ax2.set_xlabel("Hora Local del Día (VET)")
    ax2.set_ylabel("Precipitación Promedio (mm/h)")
    ax2.set_xticks(range(0, 24, 2))

    # Panel 3: Ciclo Anual de Precipitación y Temperatura
    meses_nombres = [
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic",
    ]
    ax3 = axes[1, 0]
    ax3_t = ax3.twinx()
    b1 = ax3.bar(
        anual_tp.index,
        anual_tp.values,
        color="#2ecc71",
        alpha=0.7,
        label="Precipitación Mensual (mm)",
    )
    l3 = ax3_t.plot(
        anual_t2m.index,
        anual_t2m.values,
        color="#c0392b",
        lw=2.5,
        marker="o",
        label="Temperatura Media (°C)",
    )
    ax3.set_title(
        "Ciclo Anual / Estacionalidad: Lluvias y Temperatura", fontweight="bold"
    )
    ax3.set_xlabel("Mes del Año")
    ax3.set_ylabel("Precipitación Mensual (mm)", color="#2ecc71")
    ax3_t.set_ylabel("Temperatura Media (°C)", color="#c0392b")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(meses_nombres)

    # Panel 4: Ciclo Anual de Humedad de Suelo y Radiación Solar
    ax4 = axes[1, 1]
    ax4_rad = ax4.twinx()
    l4 = ax4.plot(
        anual_swvl.index,
        anual_swvl.values,
        color="#8e44ad",
        lw=2.5,
        marker="s",
        label="Humedad Suelo (m³/m³)",
    )
    l5 = ax4_rad.plot(
        anual_ssrd.index,
        anual_ssrd.values,
        color="#f39c12",
        lw=2.5,
        linestyle=":",
        marker="^",
        label="Radiación Solar (W/m²)",
    )
    ax4.set_title("Ciclo Anual: Humedad de Suelo y Radiación Solar", fontweight="bold")
    ax4.set_xlabel("Mes del Año")
    ax4.set_ylabel("Humedad Suelo Nivel 1 (m³/m³)", color="#8e44ad")
    ax4_rad.set_ylabel("Radiación Solar SSRD (W/m²)", color="#f39c12")
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(meses_nombres)

    plt.tight_layout()
    fig_ciclos_path = "output/figures/ciclos_temporales_merida.png"
    plt.savefig(fig_ciclos_path, dpi=300)
    plt.close()
    print(f"  -> Gráfico de ciclos temporales guardado en: '{fig_ciclos_path}'")

    # -------------------------------------------------------------------
    # PARTE 2: ANÁLISIS DE FUNCIONES ORTOGONALES EMPÍRICAS (EOF / PCA)
    # -------------------------------------------------------------------
    print(
        "\n--- 2. PROCESANDO FUNCIONES ORTOGONALES EMPÍRICAS (EOF / PCA ESPACIAL) ---"
    )
    df_d = pd.read_parquet(path_diario)
    df_d["date"] = pd.to_datetime(df_d["date"])

    # Excluir píxeles del Lago de Maracaibo (16 píxeles con NaNs constantes)
    df_valid = df_d.dropna(subset=["t2m_mean", "tp_daily_sum"]).copy()

    # Matriz Pivoteada: Filas = Fechas, Columnas = Píxeles Espaciales (latitude, longitude)
    pivot_tp = df_valid.pivot(
        index="date", columns=["latitude", "longitude"], values="tp_daily_sum"
    )
    pivot_t2m = df_valid.pivot(
        index="date", columns=["latitude", "longitude"], values="t2m_mean"
    )

    # Estandarizar anomalías espacio-temporales
    tp_anom = (pivot_tp - pivot_tp.mean()) / pivot_tp.std()
    t2m_anom = (pivot_t2m - pivot_t2m.mean()) / pivot_t2m.std()

    # Ejecutar PCA / EOF para Precipitación
    pca_tp = PCA(n_components=3)
    pcs_tp = pca_tp.fit_transform(tp_anom)
    eofs_tp = pca_tp.components_  # Patrones espaciales (3, N_pixeles)
    var_exp_tp = pca_tp.explained_variance_ratio_ * 100.0

    print(f"Varianza explicada por los 3 primeros modos EOF (Precipitación):")
    print(f"  - Modo 1 (EOF1): {var_exp_tp[0]:.2f}%")
    print(f"  - Modo 2 (EOF2): {var_exp_tp[1]:.2f}%")
    print(f"  - Modo 3 (EOF3): {var_exp_tp[2]:.2f}%")
    print(f"  - Varianza Acumulada (EOF1-3): {var_exp_tp[:3].sum():.2f}%")

    # Reconstruir cuadrícula espacial 19x17 para los patrones de EOF
    unique_lats = sorted(df_d["latitude"].unique(), reverse=True)
    unique_lons = sorted(df_d["longitude"].unique())
    grid_shape = (len(unique_lats), len(unique_lons))
    lat_map = {lat: idx for idx, lat in enumerate(unique_lats)}
    lon_map = {lon: idx for idx, lon in enumerate(unique_lons)}

    # Crear mapas 2D para EOF1, EOF2, EOF3
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols_tuples = pivot_tp.columns

    for mode in range(3):
        grid_eof = np.full(grid_shape, np.nan)
        for idx_col, (lat, lon) in enumerate(cols_tuples):
            r = lat_map[lat]
            c = lon_map[lon]
            grid_eof[r, c] = eofs_tp[mode, idx_col]

        im = axes[mode].imshow(
            grid_eof,
            cmap="RdBu_r",
            extent=[
                min(unique_lons),
                max(unique_lons),
                min(unique_lats),
                max(unique_lats),
            ],
            aspect="auto",
        )
        axes[mode].set_title(
            f"Modo Spatial EOF{mode + 1} ({var_exp_tp[mode]:.1f}% Varianza)",
            fontweight="bold",
        )
        axes[mode].set_xlabel("Longitud (°W)")
        axes[mode].set_ylabel("Latitud (°N)")
        fig.colorbar(im, ax=axes[mode], label="Amplitud Normalizada")

    plt.tight_layout()
    fig_eof_spatial_path = "output/figures/eof_patrones_espaciales.png"
    plt.savefig(fig_eof_spatial_path, dpi=300)
    plt.close()
    print(
        f"  -> Gráfico de patrones espaciales EOF guardado en: '{fig_eof_spatial_path}'"
    )

    # Graficar Componentes Principales Temporales (PCs)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    dates_idx = pivot_tp.index
    colors = ["#2980b9", "#e67e22", "#2ecc71"]

    for mode in range(3):
        axes[mode].plot(
            dates_idx,
            pcs_tp[:, mode],
            color=colors[mode],
            lw=1.2,
            label=f"PC{mode + 1} ({var_exp_tp[mode]:.1f}%)",
        )
        axes[mode].axhline(0, color="black", lw=0.8, linestyle="--")
        axes[mode].set_ylabel(f"PC{mode + 1} Amplitud")
        axes[mode].legend(loc="upper right")

    axes[0].set_title(
        "Series Temporales de Componentes Principales (PCs - Precipitación Mérida)",
        fontweight="bold",
    )
    axes[2].set_xlabel("Fecha (2020 - 2026)")
    plt.tight_layout()
    fig_eof_pcs_path = "output/figures/eof_pcs_temporales.png"
    plt.savefig(fig_eof_pcs_path, dpi=300)
    plt.close()
    print(f"  -> Gráfico de series temporales de PCs guardado en: '{fig_eof_pcs_path}'")

    # -------------------------------------------------------------------
    # PARTE 3: MATRIZ DE CORRELACIÓN MULTIVARIABLE
    # -------------------------------------------------------------------
    print(
        "\n--- 3. CALCULANDO MATRIZ DE CORRELACIÓN MULTIVARIABLE Y MULTICOLINEALIDAD ---"
    )

    # Promedio espacial por fecha para todas las variables diarias
    cols_var_diarias = [
        c for c in df_valid.columns if c not in ["date", "latitude", "longitude"]
    ]
    df_mean_daily = df_valid.groupby("date")[cols_var_diarias].mean()

    # Seleccionar variables clave representativas para el mapa de calor
    vars_clave = [
        "tp_daily_sum",
        "t2m_mean",
        "t2m_min",
        "t2m_max",
        "d2m_mean",
        "swvl1_mean",
        "swvl2_mean",
        "swvl3_mean",
        "sp_mean",
        "u10_mean",
        "v10_mean",
        "ssrd_mean",
        "strd_mean",
        "sshf_mean",
        "slhf_mean",
        "e_mean",
        "evavt_mean",
    ]
    vars_presentes = [v for v in vars_clave if v in df_mean_daily.columns]

    corr_pearson = df_mean_daily[vars_presentes].corr(method="pearson")
    corr_spearman = df_mean_daily[vars_presentes].corr(method="spearman")

    # Graficar Matriz de Correlación de Pearson
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr_pearson, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(vars_presentes)))
    ax.set_yticks(range(len(vars_presentes)))
    ax.set_xticklabels(vars_presentes, rotation=90, ha="center")
    ax.set_yticklabels(vars_presentes)
    ax.set_title(
        "Matriz de Correlación de Pearson (Variables Climáticas Mérida)",
        fontweight="bold",
        pad=20,
    )

    # Anotar valores significativos en las celdas
    for i in range(len(vars_presentes)):
        for j in range(len(vars_presentes)):
            val = corr_pearson.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.7 else "white",
                fontsize=8,
            )

    plt.tight_layout()
    fig_corr_path = "output/figures/matriz_correlacion_multivariable.png"
    plt.savefig(fig_corr_path, dpi=300)
    plt.close()
    print(f"  -> Gráfico de matriz de correlación guardado en: '{fig_corr_path}'")

    # Diagnóstico de Multicolinealidad (|r| > 0.85)
    print("\n--- Diagnóstico de Multicolinealidad (|r| > 0.85) ---")
    redundancias = []
    for i in range(len(vars_presentes)):
        for j in range(i + 1, len(vars_presentes)):
            val_p = corr_pearson.iloc[i, j]
            if abs(val_p) > 0.85:
                v1, v2 = vars_presentes[i], vars_presentes[j]
                redundancias.append((v1, v2, val_p))
                print(f"  - Alta correlación entre '{v1}' y '{v2}': r = {val_p:.3f}")

    print(
        f"\n¡Análisis EDA finalizado con éxito! Todos los gráficos y diagnósticos fueron generados."
    )


if __name__ == "__main__":
    ejecutar_bloque3_eda()
