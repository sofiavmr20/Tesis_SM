# %%

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# %%

# Configuración de estilos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
print(f"Directorio de datos procesados: {DATA_DIR.resolve()}")
print(f"Directorio de resultados: {RESULTS_DIR.resolve()}")

def ejecutar_clustering_microclimatico(
    archivo_diario="era5_land_merida_diario_curado.parquet",
    archivo_horario="era5_land_merida_horario_curado.parquet"
):
    path_d = DATA_DIR / Path(archivo_diario)
    path_h = DATA_DIR / Path(archivo_horario)

    if not path_d.exists():
        print(f"Error: No se encuentra el dataset diario '{archivo_diario}'")
        return

    print("==================================================================")
    print(" BLOQUE: DIVISIÓN POR CONGLOMERADOS Y CLUSTERING MICROCLIMÁTICO")
    print("==================================================================")
    print("1. Cargando dataset diario curado de Mérida...")
    t0 = time.time()
    df_d = pd.read_parquet(path_d)
    print(f"Dataset cargado ({len(df_d):,} filas) en {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------------
    # 1. MATRIZ DE CARACTERIZACIÓN CLIMÁTICA ESPACIAL
    # -------------------------------------------------------------------
    print("\n2. Construyendo matriz de caracterización climatológica por píxel espacial...")
    
    grouped = df_d.groupby(["latitude", "longitude"])
    
    df_space = pd.DataFrame({
        "sp_mean": grouped["sp_mean"].mean(),                     # Proxy de altitud / presión Pa
        "t2m_mean": grouped["t2m_mean"].mean() - 273.15,           # K a °C
        "t2m_min": grouped["t2m_min"].min() - 273.15,
        "t2m_max": grouped["t2m_max"].max() - 273.15,
        "t2m_range": grouped["t2m_max"].max() - grouped["t2m_min"].min(), # Oscilación térmica
        "tp_daily_mean": grouped["tp_daily_sum"].mean() * 1000.0,   # m a mm/día
        "swvl1_mean": grouped["swvl1_mean"].mean(),               # Humedad de suelo
        "d2m_mean": grouped["d2m_mean"].mean() - 273.15,           # Punto de rocío °C
        "ssrd_mean": grouped["ssrd_mean"].mean() / 3600.0,         # Radiación W/m2
        "wind_speed": np.sqrt(grouped["u10_mean"].mean()**2 + grouped["v10_mean"].mean()**2)
    }).reset_index()

    # Excluir los 16 píxeles lacustres (Lago de Maracaibo) que contienen NaNs
    df_land = df_space.dropna().copy()
    print(f"Total de píxeles analizados: {len(df_space)} (Continentales: {len(df_land)}, Lacustres excluidos: {len(df_space)-len(df_land)})")

    # Features para el clustering
    feature_cols = [
        "sp_mean", "t2m_mean", "t2m_range", "tp_daily_mean", 
        "swvl1_mean", "d2m_mean", "ssrd_mean", "wind_speed"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_land[feature_cols])

    # -------------------------------------------------------------------
    # 2. EVALUACIÓN Y SELECCIÓN DEL NÚMERO DE CLUSTERS (K)
    # -------------------------------------------------------------------
    print("\n3. Evaluando métricas de validación para K = 3, 4, 5, 6...")
    k_values = [3, 4, 5, 6]
    inertias = []
    silhouettes = []
    davies_bouldin = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        
        silhouettes.append(sil)
        davies_bouldin.append(db)
        print(f"  - K = {k}: Silueta = {sil:.4f} | Davies-Bouldin = {db:.4f} | Inercia = {kmeans.inertia_:.1f}")

    # Seleccionar K=4 para óptimo balance entre métrica matemática y riqueza geográfica de Mérida
    k_optimo = 4
    print(f"\n-> Número de conglomerados seleccionado: K = {k_optimo} (Microclimas representativos de Mérida)")

    # Graficar Métricas de Validación
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(k_values, inertias, marker='o', color='#2980b9', lw=2)
    axes[0].set_title('Método del Codo (Inercia)', fontweight='bold')
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inercia')

    axes[1].plot(k_values, silhouettes, marker='s', color='#2ecc71', lw=2)
    axes[1].axvline(k_optimo, color='red', linestyle='--', label=f'Seleccionado K={k_optimo}')
    axes[1].set_title('Coeficiente de Silueta', fontweight='bold')
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()

    axes[2].plot(k_values, davies_bouldin, marker='^', color='#e74c3c', lw=2)
    axes[2].set_title('Índice Davies-Bouldin', fontweight='bold')
    axes[2].set_xlabel('Número de Clusters (K)')
    axes[2].set_ylabel('DB Index')

    plt.tight_layout()
    fig_k_path = RESULTS_DIR / "clustering_metricas_k.png"
    plt.savefig(fig_k_path, dpi=300)
    plt.close()
    print(f"  -> Gráfico de métricas de validación K guardado en: '{fig_k_path}'")

    # -------------------------------------------------------------------
    # 3. ENTRENAMIENTO FINAL Y ASIGNACIÓN DE MICROCLIMAS
    # -------------------------------------------------------------------
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=15)
    df_land["cluster_id"] = kmeans_final.fit_predict(X_scaled)

    # Ordenar clusters por temperatura media ascendente (Páramo -> Templado -> Cálido Húmedo -> Planicie)
    cluster_means = df_land.groupby("cluster_id")["t2m_mean"].mean().sort_values()
    rank_map = {old_id: new_rank for new_rank, old_id in enumerate(cluster_means.index, 1)}
    df_land["cluster_id"] = df_land["cluster_id"].map(rank_map)

    # Nombres geográficos y descriptivos de los microclimas de Mérida
    nombres_microclimas = {
        1: "C1: Páramo / Alta Montaña (Muy Frío - Seco)",
        2: "C2: Valles Interandinos (Templado - Moderado)",
        3: "C3: Piedemonte Sur / Transición a Llanos (Cálido Húmedo)",
        4: "C4: Planicie Norte / Transición al Lago (Cálido Muy Húmedo)"
    }
    
    df_land["microclima_nombre"] = df_land["cluster_id"].map(
        lambda cid: nombres_microclimas.get(cid, f"C{cid}: Microclima Zona {cid}")
    )

    # Combinar con los píxeles lacustres
    df_space_final = df_space.merge(
        df_land[["latitude", "longitude", "cluster_id", "microclima_nombre"]],
        on=["latitude", "longitude"],
        how="left"
    )
    df_space_final["cluster_id"] = df_space_final["cluster_id"].fillna(-1).astype(int)
    df_space_final["microclima_nombre"] = df_space_final["microclima_nombre"].fillna("C0: Cuerpo de Agua (Lago de Maracaibo)")

    print("\n--- Perfil Característico de los Conglomerados Microclimáticos ---")
    perfil = df_land.groupby(["cluster_id", "microclima_nombre"])[
        ["t2m_mean", "tp_daily_mean", "sp_mean", "swvl1_mean", "wind_speed"]
    ].mean()
    print(perfil)

    # -------------------------------------------------------------------
    # 4. VISUALIZACIÓN ESPACIAL DE LOS CONGLOMERADOS
    # -------------------------------------------------------------------
    print("\n4. Generando mapas espaciales y perfiles microclimáticos...")
    unique_lats = sorted(df_space["latitude"].unique(), reverse=True)
    unique_lons = sorted(df_space["longitude"].unique())
    grid_shape = (len(unique_lats), len(unique_lons))
    lat_map = {lat: idx for idx, lat in enumerate(unique_lats)}
    lon_map = {lon: idx for idx, lon in enumerate(unique_lons)}

    grid_cluster = np.full(grid_shape, np.nan)
    for _, row in df_space_final.iterrows():
        r = lat_map[row["latitude"]]
        c = lon_map[row["longitude"]]
        grid_cluster[r, c] = row["cluster_id"]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('Spectral_r', k_optimo + 1)
    im = ax.imshow(
        grid_cluster, 
        cmap=cmap, 
        extent=[min(unique_lons), max(unique_lons), min(unique_lats), max(unique_lats)],
        aspect='auto',
        vmin=-1.5,
        vmax=k_optimo + 0.5
    )

    cbar = fig.colorbar(im, ax=ax)
    ticks_pos = [-1] + list(range(1, k_optimo + 1))
    labels_txt = ["C0: Lago / Agua"] + [nombres_microclimas[i] for i in range(1, k_optimo + 1)]
    cbar.set_ticks(ticks_pos)
    cbar.set_ticklabels(labels_txt)

    ax.set_title('Zonificación Microclimática por Conglomerados (Estado Mérida)', fontweight='bold')
    ax.set_xlabel('Longitud (°W)')
    ax.set_ylabel('Latitud (°N)')

    plt.tight_layout()
    fig_map_path = RESULTS_DIR / "mapa_microclimas_merida.png"
    plt.savefig(fig_map_path, dpi=300)
    plt.close()
    print(f"  -> Mapa espacial de microclimas guardado en: '{fig_map_path}'")

    # -------------------------------------------------------------------
    # 5. INTEGRACIÓN Y GUARDADO DE CONGLOMERADOS EN LOS DATASETS CURADOS
    # -------------------------------------------------------------------
    print("\n5. Guardando la tabla de conglomerados e integrándola en los datasets curados...")
    
    path_map_parquet = DATA_DIR / "conglomerados_microclimas_merida.parquet"
    path_map_csv = DATA_DIR / "conglomerados_microclimas_merida.csv"
    
    df_map_export = df_space_final[["latitude", "longitude", "cluster_id", "microclima_nombre"]]
    df_map_export.to_parquet(path_map_parquet, index=False)
    df_map_export.to_csv(path_map_csv, index=False)
    print(f"  -> Tabla de asignación espacial guardada en '{path_map_parquet}' y '{path_map_csv}'.")

    # Adjuntar cluster_id y microclima_nombre al Dataset Diario Curado
    print(f"  -> Adjuntando conglomerados a '{archivo_diario}'...")
    df_d = df_d.drop(columns=[c for c in ["cluster_id", "microclima_nombre"] if c in df_d.columns])
    df_d = df_d.merge(df_map_export, on=["latitude", "longitude"], how="left")
    df_d.to_parquet(archivo_diario, index=False, compression="snappy")

    # Adjuntar cluster_id y microclima_nombre al Dataset Horario Curado (si existe)
    if path_h.exists():
        print(f"  -> Adjuntando conglomerados a '{archivo_horario}'...")
        df_h = pd.read_parquet(path_h)
        df_h = df_h.drop(columns=[c for c in ["cluster_id", "microclima_nombre"] if c in df_h.columns])
        df_h = df_h.merge(df_map_export, on=["latitude", "longitude"], how="left")
        df_h.to_parquet(archivo_horario, index=False, compression="snappy")
        del df_h

    print(f"\n¡Proceso de clustering microclimático finalizado exitosamente en {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    ejecutar_clustering_microclimatico()

# %%
