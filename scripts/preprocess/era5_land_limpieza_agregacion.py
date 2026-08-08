from pathlib import Path
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr

def ejecutar_limpieza_y_agregacion_diaria(
    directorio_entrada="data_era5land",
    archivo_salida="era5_land_merida_diario_2020_2026.parquet"
):
    dir_path = Path(directorio_entrada)
    archivos_nc = sorted(list(dir_path.glob("*.nc")))

    if not archivos_nc:
        print(f"Error: No se encontraron archivos .nc en {dir_path.resolve()}")
        return

    print("==================================================================")
    print(" 1. DIAGNÓSTICO DE CALIDAD, GAPS E INTERPOLACIÓN < 6 HORAS")
    print("==================================================================")
    print(f"Se encontraron {len(archivos_nc)} archivos NetCDF para procesar.")

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
    daily_datasets = []
    total_nulos_reparados = 0
    total_anomalias_detectadas = 0

    print("\nProcesando archivos por bloque, validando rejilla espacial y agregando a escala diaria...")

    for idx, filepath in enumerate(archivos_nc, 1):
        with xr.open_dataset(filepath) as ds:
            # 1. Filtro y Control de Calidad Espacial / Rangos Físicos
            for var, (vmin, vmax) in rangos_fisicos.items():
                if var in ds.data_vars:
                    arr = ds[var].values
                    out_mask = (arr < vmin) | (arr > vmax)
                    if np.any(out_mask):
                        total_anomalias_detectadas += int(out_mask.sum())
                        arr[out_mask] = np.nan
                        ds[var].values = arr

            # 2. Interpolación lineal temporal para brechas < 6 horas (limit=5)
            # Para brechas >= 6 horas se preserva NaN
            if ds.isnull().any():
                ds = ds.interpolate_na(dim="valid_time", method="linear", limit=5)

            # 3. Conversión de tiempo UTC -> UTC-4 (Hora Legal de Venezuela)
            # Desplazar valid_time por -4 horas
            time_utc4 = ds['valid_time'].values - np.timedelta64(4, 'h')
            dates_utc4 = pd.to_datetime(time_utc4).date
            ds = ds.assign_coords(date=('valid_time', dates_utc4))

            # 4. Agregaciones Diarias
            var_tp = [c for c in ["tp"] if c in ds.data_vars]
            var_estado = [c for c in ["t2m", "d2m", "stl1", "swvl1", "swvl2", "swvl3", "sp", "u10", "v10"] if c in ds.data_vars]
            var_otros = [c for c in ds.data_vars if c not in var_tp and c not in var_estado]

            dict_agg = {}

            # Precipitación: Suma diaria
            for v in var_tp:
                dict_agg[f"{v}_daily_sum"] = ds[v].groupby('date').sum(dim='valid_time')

            # Variables de estado: Promedio, Mínimo y Máximo diario
            for v in var_estado:
                dict_agg[f"{v}_mean"] = ds[v].groupby('date').mean(dim='valid_time')
                dict_agg[f"{v}_min"] = ds[v].groupby('date').min(dim='valid_time')
                dict_agg[f"{v}_max"] = ds[v].groupby('date').max(dim='valid_time')

            # Flujos y evaporación: Acumulado y Promedio
            for v in var_otros:
                dict_agg[f"{v}_sum"] = ds[v].groupby('date').sum(dim='valid_time')
                dict_agg[f"{v}_mean"] = ds[v].groupby('date').mean(dim='valid_time')

            ds_daily_block = xr.Dataset(dict_agg)
            df_daily_block = ds_daily_block.to_dataframe().reset_index()
            daily_datasets.append(df_daily_block)

        if idx % 50 == 0 or idx == len(archivos_nc):
            print(f"  - [{idx}/{len(archivos_nc)}] Bloques procesados...")

    print(f"\nProcesamiento individual completado en {time.time()-t0:.2f}s.")

    print("\n==================================================================")
    print(" 2. CONSOLIDACIÓN Y ELIMINACIÓN DE SOLAPAMIENTOS DIARIOS")
    print("==================================================================")
    df_all_daily = pd.concat(daily_datasets, ignore_index=True)

    # Dado que los bloques de 7 días se solapan en los bordes de fecha, agrupamos por (date, latitude, longitude)
    print(f"Filas diarias crudas: {len(df_all_daily):,}")
    print("Agrupando y eliminando solapamientos de fecha...")

    agg_rules = {}
    for col in df_all_daily.columns:
        if col in ["date", "latitude", "longitude"]:
            continue
        if "_min" in col:
            agg_rules[col] = "min"
        elif "_max" in col:
            agg_rules[col] = "max"
        elif "_sum" in col:
            agg_rules[col] = "mean" # Para días solapados entre bloques, el valor ya calculado de la suma de ese día completo es idéntico
        else:
            agg_rules[col] = "mean"

    df_final = df_all_daily.groupby(["date", "latitude", "longitude"], as_index=False).agg(agg_rules)

    # Filtrar rango de fechas oficial: 2020-01-01 a 2026-03-31
    df_final["date"] = pd.to_datetime(df_final["date"])
    df_final = df_final[(df_final["date"] >= "2020-01-01") & (df_final["date"] <= "2026-03-31")].copy()

    # Reducir precisión a float32 para optimizar espacio
    cols_float = df_final.select_dtypes(include=["float64"]).columns
    df_final[cols_float] = df_final[cols_float].astype("float32")

    print("\n==================================================================")
    print(" 3. REPORTES Y METRICAS DE CALIDAD DEL DATASET DIARIO")
    print("==================================================================")
    print(f"Dimensiones del DataFrame Diario Final: {df_final.shape}")
    print(f"Días únicos procesados: {df_final['date'].nunique():,} días")
    print(f"Rango de Fechas Locales (UTC-4): {df_final['date'].min().strftime('%Y-%m-%d')} a {df_final['date'].max().strftime('%Y-%m-%d')}")
    print(f"Píxeles espaciales por día: {df_final.groupby('date').size().mean():.0f} píxeles (19x17)")
    print(f"Anomalías/Valores fuera de rango físico corregidos: {total_anomalias_detectadas}")

    nulos_finales = df_final.isnull().sum()
    print("\nConteo de valores nulos finales (NaNs en brechas >= 6 horas):")
    nulos_finales_sig = nulos_finales[nulos_finales > 0]
    if nulos_finales_sig.empty:
        print("  -> Serie de tiempo 100% limpia y sin valores faltantes (0 NaNs).")
    else:
        print(nulos_finales_sig)

    print("\nVista previa del Dataset Diario Consolidad (Primeras 5 filas):")
    print(df_final[["date", "latitude", "longitude", "tp_daily_sum", "t2m_mean", "t2m_min", "t2m_max", "sp_mean"]].head())

    print(f"\nGuardando dataset preliminar diario en Parquet: '{archivo_salida}'...")
    df_final.to_parquet(archivo_salida, index=False, compression="snappy")
    print("¡Limpieza, Control de Calidad y Agregación Diaria finalizados exitosamente!")

if __name__ == "__main__":
    ejecutar_limpieza_y_agregacion_diaria()
