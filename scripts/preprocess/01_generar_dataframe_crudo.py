from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr

def generar_dataframe_crudo(
    directorio_entrada="data_era5land",
    archivo_salida="datos_crudos_era5land.parquet"
):
    dir_path = Path(directorio_entrada)
    archivos_nc = sorted(list(dir_path.glob("*.nc")))

    if not archivos_nc:
        print(f"Error: No se encontraron archivos .nc en {dir_path.resolve()}")
        return

    print("==================================================================")
    print(" ETAPA 1: GENERACIÓN Y GUARDADO DEL DATAFRAME CRUDO (ERA5-LAND)")
    print("==================================================================")
    print(f"Se encontraron {len(archivos_nc)} archivos NetCDF.")

    t0 = time.time()
    lote_tamano = 25
    total_lotes = (len(archivos_nc) + lote_tamano - 1) // lote_tamano
    dataframes_crudos = []

    print("\nLeyendo archivos y convirtiendo marcas de tiempo a UTC-4 (Hora de Venezuela)...")

    for i in range(total_lotes):
        sub_files = archivos_nc[i * lote_tamano : (i + 1) * lote_tamano]
        dss = [xr.open_dataset(f) for f in sub_files]
        
        # Concatenar archivos del lote a lo largo del tiempo
        ds_batch = xr.concat(dss, dim="valid_time")
        
        # Reducir precisión a float32 para optimización de almacenamiento
        ds_batch = ds_batch.astype("float32")
        
        # Convertir a DataFrame crudo
        df_batch = ds_batch.to_dataframe().reset_index()
        
        # Limpiar columnas auxiliares de CDS si existen
        cols_to_drop = [c for c in ["number", "expver"] if c in df_batch.columns]
        if cols_to_drop:
            df_batch.drop(columns=cols_to_drop, inplace=True)

        # Ajuste de zona horaria: valid_time (UTC) -> time_utc4 (UTC-4 VET)
        df_batch.rename(columns={"valid_time": "valid_time_utc"}, inplace=True)
        df_batch["time_utc4"] = df_batch["valid_time_utc"] - pd.Timedelta(hours=4)

        dataframes_crudos.append(df_batch)

        for d in dss:
            d.close()

        print(f"  - Lote [{i+1}/{total_lotes}] procesado (Archivos {i*lote_tamano + 1} a {min((i+1)*lote_tamano, len(archivos_nc))})...")

    # Concatenación final de DataFrames crudos
    print("\nConsolidando el DataFrame crudo completo...")
    df_crudo = pd.concat(dataframes_crudos, ignore_index=True)

    # Reordenar columnas para legibilidad
    front_cols = ["valid_time_utc", "time_utc4", "latitude", "longitude"]
    other_cols = [c for c in df_crudo.columns if c not in front_cols]
    df_crudo = df_crudo[front_cols + other_cols]

    print("\n--- Resumen del Dataset Crudo ---")
    print(f"Total de registros horarios: {len(df_crudo):,}")
    print(f"Columnas ({len(df_crudo.columns)}): {list(df_crudo.columns)}")
    print(f"Rango temporal UTC:   {df_crudo['valid_time_utc'].min()} -> {df_crudo['valid_time_utc'].max()}")
    print(f"Rango temporal UTC-4: {df_crudo['time_utc4'].min()} -> {df_crudo['time_utc4'].max()}")

    print(f"\nGuardando DataFrame crudo en '{archivo_salida}'...")
    t_save = time.time()
    df_crudo.to_parquet(archivo_salida, index=False, compression="snappy")
    print(f"¡Guardado completado exitosamente en {time.time()-t_save:.2f}s! Tiempo total: {time.time()-t0:.2f}s.")

if __name__ == "__main__":
    generar_dataframe_crudo()
