from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr

def curar_y_agregar_multiescala(
    directorio_nc="data_era5land",
    archivo_horario_curado="era5_land_merida_horario_curado.parquet",
    archivo_diario_curado="era5_land_merida_diario_curado.parquet"
):
    dir_path = Path(directorio_nc)
    archivos_nc = sorted(list(dir_path.glob("*.nc")))

    if not archivos_nc:
        print(f"Error: No se encontraron archivos .nc en {dir_path.resolve()}")
        return

    print("==================================================================")
    print(" ETAPA 2: CURACIÓN DE DATOS Y EXPORTACIÓN EN DOBLE ESCALA (UTC-4)")
    print("==================================================================")
    print(f"Se procesarán {len(archivos_nc)} archivos NetCDF.")

    rangos_fisicos = {
        "t2m": (230.0, 330.0),    # K
        "d2m": (230.0, 330.0),    # K
        "stl1": (230.0, 330.0),   # K
        "tp": (0.0, 0.5),         # m/h
        "swvl1": (0.0, 1.0),      # m3/m3
        "swvl2": (0.0, 1.0),
        "swvl3": (0.0, 1.0),
        "sp": (50000.0, 110000.0),# Pa
        "u10": (-75.0, 75.0),     # m/s
        "v10": (-75.0, 75.0),     # m/s
    }

    t0 = time.time()
    dfs_horarios_curados = []
    dfs_diarios_curados = []
    total_anomalias_detectadas = 0

    print("\n1. Aplicando Control de Calidad Espacial, Filtros Físicos e Interpolación < 6h...")

    for idx, filepath in enumerate(archivos_nc, 1):
        with xr.open_dataset(filepath) as ds:
            # A. Control de Calidad Espacial / Rangos Físicos
            for var, (vmin, vmax) in rangos_fisicos.items():
                if var in ds.data_vars:
                    arr = ds[var].values
                    out_mask = (arr < vmin) | (arr > vmax)
                    if np.any(out_mask):
                        total_anomalias_detectadas += int(out_mask.sum())
                        arr[out_mask] = np.nan
                        ds[var].values = arr

            # B. Manejo de Gaps: Interpolación lineal temporal para brechas < 6h (limit=5)
            # Brechas >= 6 horas se preservan explícitamente como NaN
            if ds.isnull().any():
                ds = ds.interpolate_na(dim="valid_time", method="linear", limit=5)

            # C. Homogeneización Temporal: UTC -> UTC-4 (Hora Legal de Venezuela)
            time_utc4 = ds["valid_time"].values - np.timedelta64(4, "h")
            dates_utc4 = pd.to_datetime(time_utc4).date
            
            ds = ds.assign_coords(time_utc4=("valid_time", time_utc4))
            ds = ds.assign_coords(date=("valid_time", dates_utc4))

            # -------------------------------------------------------------
            # ESCALA 1: CONVERSIÓN Y ACUMULACIÓN HORARIA CURADA
            # -------------------------------------------------------------
            df_horario_block = ds.to_dataframe().reset_index()
            cols_drop = [c for c in ["number", "expver"] if c in df_horario_block.columns]
            if cols_drop:
                df_horario_block.drop(columns=cols_drop, inplace=True)
            
            df_horario_block.rename(columns={"valid_time": "valid_time_utc"}, inplace=True)
            dfs_horarios_curados.append(df_horario_block)

            # -------------------------------------------------------------
            # ESCALA 2: AGREGACIÓN TEMPORAL DIARIA CURADA
            # -------------------------------------------------------------
            var_tp = [c for c in ["tp"] if c in ds.data_vars]
            var_estado = [c for c in ["t2m", "d2m", "stl1", "swvl1", "swvl2", "swvl3", "sp", "u10", "v10"] if c in ds.data_vars]
            var_otros = [c for c in ds.data_vars if c not in var_tp and c not in var_estado]

            dict_agg = {}
            for v in var_tp:
                dict_agg[f"{v}_daily_sum"] = ds[v].groupby("date").sum(dim="valid_time")

            for v in var_estado:
                dict_agg[f"{v}_mean"] = ds[v].groupby("date").mean(dim="valid_time")
                dict_agg[f"{v}_min"] = ds[v].groupby("date").min(dim="valid_time")
                dict_agg[f"{v}_max"] = ds[v].groupby("date").max(dim="valid_time")

            for v in var_otros:
                dict_agg[f"{v}_sum"] = ds[v].groupby("date").sum(dim="valid_time")
                dict_agg[f"{v}_mean"] = ds[v].groupby("date").mean(dim="valid_time")

            ds_daily_block = xr.Dataset(dict_agg)
            df_daily_block = ds_daily_block.to_dataframe().reset_index()
            dfs_diarios_curados.append(df_daily_block)

        if idx % 50 == 0 or idx == len(archivos_nc):
            print(f"  - [{idx}/{len(archivos_nc)}] Bloques curados y agregados...")

    print(f"\nProcesamiento individual de bloques completado en {time.time()-t0:.2f}s.")

    # -------------------------------------------------------------
    # CONSOLIDACIÓN Y GUARDADO DE LA ESCALA HORARIA CURADA
    # -------------------------------------------------------------
    print("\n2. Consolidando y guardando Dataset Horario Curado (UTC-4)...")
    df_horario_final = pd.concat(dfs_horarios_curados, ignore_index=True)
    df_horario_final.drop_duplicates(subset=["time_utc4", "latitude", "longitude"], inplace=True)
    df_horario_final.sort_values(by=["time_utc4", "latitude", "longitude"], inplace=True)
    df_horario_final.reset_index(drop=True, inplace=True)

    cols_f32_h = df_horario_final.select_dtypes(include=["float64"]).columns
    df_horario_final[cols_f32_h] = df_horario_final[cols_f32_h].astype("float32")

    print(f"  - Registros horarios curados: {len(df_horario_final):,}")
    print(f"  - Rango de fechas UTC-4: {df_horario_final['time_utc4'].min()} -> {df_horario_final['time_utc4'].max()}")
    print(f"  - Guardando '{archivo_horario_curado}'...")
    df_horario_final.to_parquet(archivo_horario_curado, index=False, compression="snappy")
    del df_horario_final  # Liberar memoria

    # -------------------------------------------------------------
    # CONSOLIDACIÓN Y GUARDADO DE LA ESCALA DIARIA CURADA
    # -------------------------------------------------------------
    print("\n3. Consolidando y guardando Dataset Diario Curado (UTC-4)...")
    df_diario_raw = pd.concat(dfs_diarios_curados, ignore_index=True)
    
    agg_rules = {}
    for col in df_diario_raw.columns:
        if col in ["date", "latitude", "longitude"]:
            continue
        if "_min" in col:
            agg_rules[col] = "min"
        elif "_max" in col:
            agg_rules[col] = "max"
        elif "_sum" in col:
            agg_rules[col] = "mean"
        else:
            agg_rules[col] = "mean"

    df_diario_final = df_diario_raw.groupby(["date", "latitude", "longitude"], as_index=False).agg(agg_rules)
    df_diario_final["date"] = pd.to_datetime(df_diario_final["date"])
    df_diario_final = df_diario_final[(df_diario_final["date"] >= "2020-01-01") & (df_diario_final["date"] <= "2026-03-31")].copy()
    df_diario_final.sort_values(by=["date", "latitude", "longitude"], inplace=True)
    df_diario_final.reset_index(drop=True, inplace=True)

    cols_f32_d = df_diario_final.select_dtypes(include=["float64"]).columns
    df_diario_final[cols_f32_d] = df_diario_final[cols_f32_d].astype("float32")

    print(f"  - Registros diarios curados: {len(df_diario_final):,}")
    print(f"  - Días únicos procesados: {df_diario_final['date'].nunique():,} días")
    print(f"  - Guardando '{archivo_diario_curado}'...")
    df_diario_final.to_parquet(archivo_diario_curado, index=False, compression="snappy")

    print("\n==================================================================")
    print(" RESUMEN FINAL DE LA ETAPA 2 (CURACIÓN Y DOBLE ESCALA)")
    print("==================================================================")
    print(f"  1. Dataset Horario Curado: '{archivo_horario_curado}'")
    print(f"  2. Dataset Diario Curado:  '{archivo_diario_curado}'")
    print(f"  3. Anomalías/Fuera de rango corregidos: {total_anomalias_detectadas}")
    print("¡Curación y exportación multiescala finalizadas con éxito!")

if __name__ == "__main__":
    curar_y_agregar_multiescala()
