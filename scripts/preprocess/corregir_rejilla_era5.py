# %%

from pathlib import Path
import tempfile
import shutil
import numpy as np
import xarray as xr

# %%

def corregir_coordenadas_y_rejilla(directorio_era5="data_era5land"):
    dir_path = Path(directorio_era5)
    archivos_nc = sorted(list(dir_path.glob("*.nc")))
    
    if not archivos_nc:
        print(f"No se encontraron archivos .nc en {dir_path.resolve()}")
        return

    total = len(archivos_nc)
    print(f"Iniciando la corrección de coordenadas en {total} archivos .nc en '{dir_path}'...")
    
    exitosos = 0
    errores = 0

    for i, nc_file in enumerate(archivos_nc, 1):
        try:
            with xr.open_dataset(nc_file) as ds:
                # Verificar si tiene las 34 longitudes o requiere normalización
                if ds.sizes.get("longitude", 0) == 17 and (ds.longitude.values < 0).all():
                    continue

                # 1. Normalizar longitudes de [0, 360] a [-180, 180] y redondear
                lons = np.round(np.where(ds.longitude.values > 180, ds.longitude.values - 360, ds.longitude.values), 2)
                lats = np.round(ds.latitude.values, 2)

                # 2. Re-asignar coordenadas y consolidar rejilla a 19x17
                ds_norm = ds.assign_coords(longitude=lons, latitude=lats)
                ds_fixed = ds_norm.groupby("longitude").first().groupby("latitude").first()

                # Guardar temporalmente y reemplazar
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir=dir_path) as tmp:
                    tmp_name = tmp.name
                
                ds_fixed.to_netcdf(tmp_name)

            shutil.move(tmp_name, nc_file)
            exitosos += 1

            if i % 25 == 0 or i == total:
                print(f"[{i}/{total}] Corregidos {exitosos} archivos (Rejilla 19x17)...")

        except Exception as e:
            print(f"[{i}/{total}] Error al corregir {nc_file.name}: {e}")
            errores += 1

    print(f"\n¡Corrección finalizada! Archivos procesados: {exitosos}, Errores: {errores}.")

if __name__ == "__main__":
    corregir_coordenadas_y_rejilla()

# %%