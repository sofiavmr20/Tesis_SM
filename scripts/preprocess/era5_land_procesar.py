# %%

from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr


class archivos_nc(ABC):
    def __init__(
        self,
        directorio_entrada="data_era5land",
        archivo_salida="era5_land_merida_2020_2026_utc4.parquet",
        lote_tamano=25,
    ):
        self.directorio_entrada = directorio_entrada
        self.archivo_salida = archivo_salida
        self.lote_tamano = lote_tamano
        self.archivos = []

    @abstractmethod
    def listar_archivos(self):
        raise NotImplementedError

    @abstractmethod
    def inspeccionar_archivo(self, sample_file):
        raise NotImplementedError

    @abstractmethod
    def procesar_lotes(self):
        raise NotImplementedError

    @abstractmethod
    def guardar_parquet(self, df):
        raise NotImplementedError

    def procesar_era5_a_parquet(self):
        self.listar_archivos()

        print("==================================================")
        print(" INSPECCIÓN Y PROCESAMIENTO DE DATOS ERA5-LAND")
        print("==================================================")
        print(f"Se encontraron {len(self.archivos)} archivos NetCDF.")

        info = self.inspeccionar_archivo(self.archivos[0])

        print("\n--- Inspección Estructural ---")
        print(f"Coordenadas detectadas: {info['coords']}")
        print(f"Dimensiones espaciales por bloque: {info['dims']}")
        print(
            f"Extensión espacial (Mérida, Venezuela): Lat [{info['lat_range'][0]}, {info['lat_range'][1]}], "
            f"Lon [{info['lon_range'][0]}, {info['lon_range'][1]}]"
        )
        print(f"Variables registradas ({len(info['vars_list'])}): {info['vars_list']}")

        print("\n--- Procesando y Convirtiendo a UTC-4 (Hora Legal de Venezuela) ---")
        df_final = self.procesar_lotes()

        print("\nCombinando todos los DataFrames...")
        print(f"\n--- Resultado Final ---")
        print(f"Dimensiones del DataFrame consolidado: {df_final.shape}")
        print(
            f"Rango temporal UTC:   {df_final['valid_time_utc'].min()} -> {df_final['valid_time_utc'].max()}"
        )
        print(
            f"Rango temporal UTC-4: {df_final['time_utc4'].min()} -> {df_final['time_utc4'].max()}"
        )
        print("\nVista previa de las primeras filas:")
        print(df_final.head())

        print(f"\nGuardando en formato Parquet comprimido ({self.archivo_salida})...")
        df_final = self.guardar_parquet(df_final)
        print("¡Procesamiento y guardado completados exitosamente!")
        return df_final


class archivos_nc_impl(archivos_nc):
    def listar_archivos(self):
        dir_path = Path(self.directorio_entrada)
        archivos_nc = sorted(dir_path.glob("*.nc"))
        if not archivos_nc:
            raise FileNotFoundError(
                f"Error: No se encontraron archivos .nc en {dir_path.resolve()}"
            )
        self.archivos = archivos_nc
        return self.archivos

    def inspeccionar_archivo(self, sample_file):
        with xr.open_dataset(sample_file) as ds_sample:
            return {
                "coords": list(ds_sample.coords.keys()),
                "dims": dict(ds_sample.sizes),
                "vars_list": list(ds_sample.data_vars.keys()),
                "lat_range": (
                    float(ds_sample.latitude.min()),
                    float(ds_sample.latitude.max()),
                ),
                "lon_range": (
                    float(ds_sample.longitude.min()),
                    float(ds_sample.longitude.max()),
                ),
            }

    def procesar_lotes(self):
        if not self.archivos:
            self.listar_archivos()

        total_lotes = (len(self.archivos) + self.lote_tamano - 1) // self.lote_tamano
        dataframes = []

        for i in range(total_lotes):
            sub_files = self.archivos[i * self.lote_tamano : (i + 1) * self.lote_tamano]
            dss = [xr.open_dataset(f) for f in sub_files]

            ds_batch = xr.concat(dss, dim="valid_time")
            ds_batch = ds_batch.astype("float32")
            df_batch = ds_batch.to_dataframe().reset_index()

            cols_to_drop = [c for c in ["number", "expver"] if c in df_batch.columns]
            if cols_to_drop:
                df_batch.drop(columns=cols_to_drop, inplace=True)

            df_batch.rename(columns={"valid_time": "valid_time_utc"}, inplace=True)
            df_batch["time_utc4"] = df_batch["valid_time_utc"] - pd.Timedelta(hours=4)

            dataframes.append(df_batch)
            for d in dss:
                d.close()

            print(
                f"Lote [{i + 1}/{total_lotes}] procesado "
                f"(Archivos {i * self.lote_tamano + 1} a {min((i + 1) * self.lote_tamano, len(self.archivos))})..."
            )

        return pd.concat(dataframes, ignore_index=True)

    def guardar_parquet(self, df):
        front_cols = ["valid_time_utc", "time_utc4", "latitude", "longitude"]
        other_cols = [c for c in df.columns if c not in front_cols]
        df_final = df[front_cols + other_cols]
        df_final.to_parquet(self.archivo_salida, index=False, compression="snappy")
        return df_final


# %%


def procesar_era5_a_parquet(
    directorio_entrada="data_era5land",
    archivo_salida="era5_land_merida_2020_2026_utc4.parquet",
):
    processor = archivos_nc_impl(directorio_entrada, archivo_salida)
    return processor.procesar_era5_a_parquet()


if __name__ == "__main__":
    procesar_era5_a_parquet()
