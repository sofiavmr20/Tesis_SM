import glob
import os
import shutil
import h5netcdf
import xarray as xr



bbox_merida = (-72.0, 7.5, -70.4, 9.3)
local_path = "./data/GPM_IMERG"
zarr_output_path = "./data/GPM_IMERG.zarr"

print("🔍 Analizando la estructura de los archivos NetCDF4...")

all_files = sorted(glob.glob(f"{local_path}/*.nc4"))
if not all_files:
    raise FileNotFoundError(f"No se encontraron archivos .nc4 en {local_path}")

# Inspect the first valid file to detect structure
target_group = None
valid_files = []

for filepath in all_files:
    try:
        with h5netcdf.File(filepath, mode="r") as f:
            # Detect group hierarchy on the first successful read
            if target_group is None:
                if "Grid" in f.groups:
                    target_group = "Grid"
                else:
                    target_group = False  # Variables in root '/'

            # Validate remaining files match structure
            if target_group == "Grid" and "Grid" not in f.groups:
                continue

            valid_files.append(filepath)
    except Exception as e:
        print(f"⚠️ Archivo ilegible o incompleto omitido: {filepath}, {str(e)}")

group_label = f"Grupo '{target_group}'" if target_group else "Raíz ('/')"
print(f"✅ Estructura detectada: {group_label}")
print(
    f"📂 Convirtiendo {len(valid_files)} archivos válidos → Zarr (con recorte espacial) ..."
)

# Parametrize open_mfdataset based on detected group structure
open_kwargs = {
    "combine": "nested",
    "concat_dim": "time",
    "engine": "h5netcdf",
    "chunks": {"time": 100, "lon": -1, "lat": -1},
}
if target_group:
    open_kwargs["group"] = target_group

ds = xr.open_mfdataset(valid_files, **open_kwargs)

# Identify precipitation variable
precip_variable = next(
    (var for var in ["precipitationCal", "precipitation", "precip"] if var in ds),
    None,
)

if not precip_variable:
    raise KeyError(
        f"No se encontró la variable de precipitación. Variables disponibles: {list(ds.data_vars)}"
    )

precip = ds[precip_variable]

# Spatial slicing
precip_merida = precip.sel(
    lon=slice(bbox_merida[0], bbox_merida[2]),
    lat=slice(bbox_merida[1], bbox_merida[3]),
)

if os.path.exists(zarr_output_path):
    shutil.rmtree(zarr_output_path)

precip_merida.to_zarr(
    zarr_output_path,
    mode="w",
    encoding={precip_variable: {"compressor": None}},
)

print(f"🚀 Zarr guardado exitosamente en {zarr_output_path}")
