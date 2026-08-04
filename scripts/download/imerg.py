# %% [markdown]
# # Descarga de GPM IMERG (NASA GES DISC)
#
# Solo descarga: búsqueda, deduplicación, descarga paralela con reanudación.
# Los archivos netCDF4 crudos se guardan en `local_path`.
# Activar `convert_to_zarr` para también generar un dataset Zarr recortado.

# %%
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthaccess

earthaccess.login(strategy="netrc", persist=True)

# %% [markdown]
# ## Configuración

# %%
bbox_merida = (-72.0, 7.5, -70.4, 9.3)
begin = "2021-01-01"
end = "2026-03-31"
dataset_id = "GPM_3IMERGDL"
local_path = "./data/GPM_IMERG"
max_workers = 4
force_redownload = False
convert_to_zarr = True
zarr_output_path = "./data/GPM_IMERG.zarr"

os.makedirs(local_path, exist_ok=True)

# %%
print("🔍 Buscando gránulos GPM IMERG en NASA GES DISC ...")
results = earthaccess.search_data(
    short_name=dataset_id,
    bounding_box=bbox_merida,
    temporal=(begin, end),
)
print(f"🛰  {len(results)} gránulos encontrados.")

# %%
granule_by_name = {}
for g in results:
    name = os.path.basename(g.data_links()[0])
    granule_by_name[name] = g
all_granules = list(granule_by_name.values())
print(f"🛰  {len(all_granules)} gránulos únicos tras deduplicación.")

if force_redownload:
    pending = all_granules
    skipped = 0
else:
    existing = set(os.listdir(local_path))
    pending = [
        g for g in all_granules if os.path.basename(g.data_links()[0]) not in existing
    ]
    skipped = len(all_granules) - len(pending)
    if skipped:
        print(f"⏭️  {skipped} gránulos ya descargados — se omiten.")

# %%
if pending:
    print(f"⬇️  Descargando {len(pending)} gránulos ({max_workers} hilos) ...")

    def _download_one(granule):
        name = os.path.basename(granule.data_links()[0])
        try:
            earthaccess.download(granule, local_path=local_path)
            return name, True, None
        except Exception as e:
            return name, False, str(e)

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, g): g for g in pending}
        try:
            from tqdm import tqdm

            iterator = tqdm(
                as_completed(futures), total=len(futures), desc="Descargando"
            )
        except ImportError:
            iterator = as_completed(futures)
        for future in iterator:
            name, ok, err = future.result()
            if ok:
                completed += 1
            else:
                failed += 1
                print(f"  ⚠️  Falló: {name}: {err}")

    print(
        f"✅ Completados: {completed} | Omitidos (caché): {skipped} | Fallidos: {failed}"
    )
else:
    print("✅ Todos los gránulos ya estaban descargados.")

# %%
