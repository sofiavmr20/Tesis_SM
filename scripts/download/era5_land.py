# %%

from math import ceil
import calendar
import datetime
from pathlib import Path
import cdsapi
import xarray as xr

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "soil_temperature_level_1",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "evaporation_from_bare_soil",
        "evaporation_from_open_water_surfaces_excluding_oceans",
        "evaporation_from_the_top_of_canopy",
        "evaporation_from_vegetation_transpiration",
        "total_evaporation",
        "sub_surface_runoff",
        "surface_runoff",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation",
    ],
    "year": "2020",
    "month": "01",
    "day": [],  # Se llena dinámicamente en el bucle
    "time": [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
    ],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [9.3, -72, 7.5, -70.4],
}

# %%

def get_begin(block_size):
    nc_files = list(Path.cwd().glob("*.nc"))
    if not nc_files:
        most_recent = None
    else:
        # Busca el archivo modificado más recientemente para retomar desde ahí
        most_recent = max(nc_files, key=lambda p: p.stat().st_mtime)
        
    if most_recent:
        stem = most_recent.stem
        partes = stem.split("_")
        year = partes[3]
        month = partes[4]
        block = partes[5][1:]  # Quita la 'b' de 'b1', 'b2', etc.
        
        # Calcula el día aproximado donde quedó el último bloque
        day = (int(block) - 1) * block_size + 1
        return datetime.datetime(int(year), int(month), day)
    else:
        # Fecha base de inicio si la carpeta está vacía
        return datetime.datetime(2020, 1, 1)

# %%

client = cdsapi.Client()

# %%

block_size = 7

begin = get_begin(block_size)
end = datetime.datetime(2026, 3, 31)

# %%
current = begin
while current <= end:
    year_str = f"{current.year}"
    month_str = f"{current.month:02d}"
    request["year"] = year_str
    request["month"] = month_str

    _, last_day = calendar.monthrange(current.year, current.month)
    n_blocks = ceil(last_day / block_size)
    
    block_days = [
        [
            f"{d:02d}"
            for d in range(i * block_size + 1, i * block_size + block_size + 1)
            if d <= last_day
        ]
        for i in range(n_blocks)
    ]

    for i, bloque in enumerate(block_days, start=1):
        # Control de reanudación: si el bloque entero pertenece a días ya procesados, se salta
        if int(bloque[-1]) < current.day:
            continue

        request["day"] = bloque
        fname = f"era5_land_merida_{year_str}_{month_str}_b{i}.nc"
        
        print(
            f"Solicitando {year_str}-{month_str} | Bloque {i} (Días {bloque[0]} al {bloque[-1]})..."
        )
        
        try:
            client.retrieve(dataset, request).download(fname)
        except Exception as e:
            print(f"Error al descargar el bloque {i} de {year_str}-{month_str}: {e}")
            raise e

    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1, day=1)
    else:
        current = current.replace(month=current.month + 1, day=1)

print("Descarga completada con éxito")

# %%

print("\nIniciando la conversión de archivos NetCDF a DataFrame...")

archivos_nc = sorted(list(Path.cwd().glob("*.nc")))

if not archivos_nc:
    print("No se encontraron archivos .nc para procesar.")
else:
    print(f"Se encontraron {len(archivos_nc)} archivos NetCDF. Combinando...")
    
    ds = xr.open_mfdataset(archivos_nc, combine="by_coords", chunks={"time": 744})
    
    print("Estructura de dimensiones leída correctamente:")

    print(ds)

    print("Transformando a Pandas DataFrame (esto puede tardar unos minutos)...")
    df = ds.to_dataframe()
    
    df = df.reset_index()
    
    print(f"\n¡Conversión exitosa! Dimensiones del DataFrame final: {df.shape}")
    
    print(df.head())
    
    output_path = "era5_land_merida_2020_2026.parquet"
    print(f"Guardando DataFrame optimizado en: {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Guardado completo.")