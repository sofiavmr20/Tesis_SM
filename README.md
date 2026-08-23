# Modelos de series de tiempo espacio-temporales para predicción meteorológica

Este repositorio contiene el código, datos de ejemplo y documentación asociada al trabajo de grado:

**"Modelos de series de tiempo espacio-temporales para la predicción de variables meteorológicas orientados a alertas tempranas de crecidas de ríos en la cuenca del río Chama, estado Mérida"**

## 📋 Descripción del proyecto

El objetivo principal es desarrollar y comparar modelos de series de tiempo que incorporen dependencias espaciales y temporales para predecir precipitación, temperatura y nivel de ríos, utilizando datos de estaciones meteorológicas terrestres e imágenes satelitales de acceso abierto. Los resultados buscan mejorar los sistemas de alerta temprana por crecidas súbitas en los Andes venezolanos.

## 🎯 Objetivos

- Recopilar y armonizar datos diarios (2015–2024) de estaciones meteorológicas y satélites.
- Implementar modelos tradicionales: SARIMA y VAR.
- Implementar modelos espacio-temporales avanzados, jerárquico bayesiano y ConvLSTM.
- Evaluar el desempeño predictivo y seleccionar el mejor modelo.
- Proponer umbrales de alerta calibrados con eventos históricos de crecidas.

## 📁 Estructura del repositorio

```
├── data/
│   ├── raw/                  # Datos originales (no versionados)
│   ├── processed/            # Datos armonizados y listos para modelado
│   ├── spatial/              # Shapefiles, matrices de pesos espaciales
│   ├── GPM_IMERG/            # Datos satelitales GPM IMERG (diarios)
│   └── GPM_IMERG.zarr/       # Cubo espacio-temporal en formato Zarr
├── notebooks/                # Jupyter notebooks y RMarkdown
├── scripts/
│   ├── download/             # Descarga de datos (GPM IMERG, ERA5)
│   ├── preprocess/           # Limpieza, imputación, control de calidad
│   ├── models/               # StatsForecast, MLForecast, NeuralForecast
│   ├── evaluation/           # Métricas de error, pruebas estadísticas
│   └── exploratory/          # Scripts de prueba y exploración
├── output/                   # Figuras y gráficos generados
│   └── figures/
├── results/                  # Resultados finales
│   ├── figures/
│   ├── tables/
│   └── models/
├── docs/                     # Documentación (informe, presentación)
├── pyproject.toml            # Dependencias de Python (uv)
├── environment.yml           # Entorno Conda alternativo
├── LICENSE                   # Licencia del proyecto
└── README.md                 # Este archivo
```

## 🛠️ Requerimientos de software y librerías

### Python
Las principales librerías se listan en `pyproject.toml`. Instalación:

```bash
uv sync --upgrade
```

## 🚀 Instrucciones de uso (flujo de trabajo)

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/sofiavmr20/Tesis_SM.git
   cd Tesis_SM
   ```

2. **Descargar datos**  
   - Solicitar datos de estaciones
   - Descargar satelitales usando scripts en `scripts/download`.

3. **Preprocesar datos**  
   Ejecutar el pipeline de control de calidad e imputación:

   ```bash
   python scripts/preprocess/quality_control.py
   ```

4. **Análisis exploratorio** 

   Abrir `notebooks/01_eda.ipynb` (Jupyter).

5. **Ejecutar modelos**  

   - AutoARIMA: `notebooks/02_statsforecast.ipynb`  

6. **Evaluar y comparar**  

   `python scripts/evaluation/compare_models.py`  
   Genera tablas de RMSE, MAE, CSI y la prueba de Diebold-Mariano.

7. **Generar propuesta de alerta**  

   `Rscript scripts/evaluation/threshold_alerts.R`  
   Los resultados se guardan en `results/best_model/`.

## 📊 Conjuntos de datos utilizados

| Fuente | Variables | Resolución | Periodo |
|--------|-----------|------------|---------|
| Estaciones | Precipitación, temperatura, nivel de río | Diaria | 2015–2024 |
| GPM-IMERG (NASA) | Precipitación | 0.1°, diaria | 2015–2024 |
| ERA5 (Copernicus) | Humedad, presión, temperatura | 0.25°, horaria → diaria | 2015–2024 |

### Datos adicionales

Los datos de las divisiones administrativas se obtuvieron de GADM (Database of Global Administrative Areas - gadm.org). Se descargó el mapa a nivel ADM2 que en el caso de Venezuela corresponde a Municipios.

- https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_VEN.gpkg

Tiene como sistema de coordenadas de referencia latitud y longitud, y datum WGS84.

El modelo digital de terreno se descargó de OpenTopography, seleccionando el producto "Copernicus GLO-30 Digital Elevation Model" https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3, introduciendo las coordenadas manualmente: 
   Xmin: -72.1; Ymin: 7.4;  Xmax: -70.3; Ymax: 9.4

Esto es, un decimal adicional en cada dirección sobre los datos extraídos de Era5-Land)

El archivo descargado se reproyectó a UTM y de este se extrajo la pendiente:

```bash
$ gdalwarp \
  -t_srs EPSG:32619 \
  -tr 30 30 \
  -r bilinear \
  -multi \
  merida_dem_wgs84.tif \
  merida_dem_utm30.tif
$ gdaldem slope \
  -compute_edges \
  merida_dem_utm30.tif \
  merida_slope_deg.tif
$ gdaldem slope \
  -p \
  -compute_edges \
  merida_dem_utm30.tif \
  merida_slope_pct.tif
``

**Nota:** Los datos brutos no se incluyen en este repositorio por políticas de acceso. Los scripts de descarga y ejemplos sintéticos se proporcionan para reproducibilidad.

## 📈 Resultados esperados

- Comparación cuantitativa de modelos con mejora ≥15% en RMSE para eventos extremos.
- Modelo final (bayesiano o ConvLSTM) calibrado para la cuenca del río Chama.
- Propuesta operativa de umbrales de alerta temprana (percentiles dinámicos) con matriz de confusión.

## 🤝 Contribuciones

Este repositorio es parte de un trabajo de pregrado. No se aceptan contribuciones externas durante el desarrollo, pero se agradecen sugerencias vía issues.

## 📄 Licencia

Este proyecto está bajo la licencia Creative Commons BY-NC-SA 4.0.

## 📧 Contacto

- **Estudiante:** Sofía Mercado – sofiavmr20@gmail.com
- **Tutor:** Francisco Palm – fpalm@qu4nt.com  
- **Línea de investigación:** Estadística aplicada, modelado espacio-temporal, hidroclimatología

---

*Desarrollado en el marco del proyecto de grado para la Universided de Los Andes, Mérida, Venezuela.*
