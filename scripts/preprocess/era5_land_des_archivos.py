# %%

from pathlib import Path
import os
import tempfile
import zipfile
import xarray as xr

# %%


def descomprimir_y_fusionar_era5(directorio_era5="data_era5land", borrar_zip=False):
    dir_path = Path(directorio_era5)
    archivos_zip = sorted(list(dir_path.glob("*.zip")))

    if not archivos_zip:
        print(f"No se encontraron archivos .zip en {dir_path.resolve()}")
        return

    total = len(archivos_zip)
    print(f"Iniciando el procesamiento de {total} archivos .zip en '{dir_path}'...")

    exitosos = 0
    errores = 0

    for i, zip_file in enumerate(archivos_zip, 1):
        target_nc = zip_file.with_suffix(".nc")

        # Si el .nc ya existe, saltar
        if target_nc.exists():
            if borrar_zip and zip_file.exists():
                zip_file.unlink()
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(tmpdir)

                nc_internos = [
                    os.path.join(tmpdir, f)
                    for f in os.listdir(tmpdir)
                    if f.endswith(".nc")
                ]

                if not nc_internos:
                    print(
                        f"[{i}/{total}] Advertencia: No hay archivos .nc dentro de {zip_file.name}"
                    )
                    continue

                # Cargar y fusionar sub-archivos sin depender de Dask
                dss = [xr.open_dataset(f) for f in nc_internos]
                with xr.merge(dss, compat="override") as ds_merged:
                    ds_merged.to_netcdf(target_nc)

                # Liberar descriptores de archivo para evitar errores de permisos en Windows
                for ds in dss:
                    ds.close()

            exitosos += 1
            if borrar_zip:
                zip_file.unlink()

            if i % 25 == 0 or i == total:
                print(f"[{i}/{total}] Procesados {exitosos} bloques correctamente...")

        except Exception as e:
            print(f"[{i}/{total}] Error al procesar {zip_file.name}: {e}")
            errores += 1

    print(f"\n¡Proceso finalizado! Éxitos: {exitosos}, Errores: {errores}.")


if __name__ == "__main__":
    descomprimir_y_fusionar_era5()

# %%
