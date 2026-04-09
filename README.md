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
├── data/                     # Datos (raw, procesados, imputados)
│   ├── raw/                  # Datos originales (no versionados si son grandes)
│   ├── processed/            # Datos armonizados y listos para modelado
│   └── spatial/              # Shapefiles, matrices de pesos espaciales
├── notebooks/                # Jupyter notebooks y RMarkdown
│   ├── 01_eda.Rmd            # Análisis exploratorio
│   ├── 02_sarima_var.ipynb   # Modelos tradicionales
│   └── 03_conv_lstm.ipynb    # ConvLSTM con datos satelitales
├── scripts/                  # Scripts reutilizables (R y Python)
│   ├── preprocess/           # Limpieza, imputación, control de calidad
│   ├── models/               # Funciones para SARIMA, VAR, STAR, INLA, ConvLSTM
│   └── evaluation/           # Métricas de error, pruebas estadísticas
├── results/                  # Figuras, tablas, métricas comparativas
│   ├── figures/
│   ├── tables/
│   └── best_model/           # Modelo final y umbrales de alerta
├── docs/                     # Documentación adicional (informe, presentación)
├── requirements.txt          # Dependencias de Python
├── renv.lock / renv/         # (Opcional) Entorno reproducible para R
├── LICENSE                   # Licencia del proyecto
└── README.md                 # Este archivo
```

## 🛠️ Requerimientos de software y librerías

### Python
Las principales librerías se listan en `pyproject.toml`. Instalación:

```bash
uv sync --upgrade
```

### R (4.0+)
Para los modelos STAR, INLA y análisis espacio-temporal:

```r
install.packages(c("tidyverse", "sf", "raster", "spacetime", "gstat", 
                   "forecast", "vars", "tseries", "ggplot2", "tmap"))
# INLA se instala desde repositorio propio:
install.packages("INLA", repos="https://inla.r-inla-download.org/R/stable")
```

## 🚀 Instrucciones de uso (flujo de trabajo)

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/sofiavmr20/Tesis_SM.git
   cd Tesis_SM
   ```

2. **Descargar datos**  
   - Solicitar datos de estaciones
   - Descargar satelitales usando scripts en `scripts/preprocess/download_sat_data.py`.

3. **Preprocesar datos**  
   Ejecutar el pipeline de control de calidad e imputación:

   ```bash
   python scripts/preprocess/quality_control.py
   Rscript scripts/preprocess/impute_spatial.R
   ```

4. **Análisis exploratorio** 

   Abrir `notebooks/01_eda.Rmd` (RStudio) o `notebooks/01_eda.ipynb` (Jupyter).

5. **Ejecutar modelos**  

   - SARIMA / VAR: `notebooks/02_sarima_var.ipynb`  
   - STAR(1,1): `Rscript scripts/models/star_model.R`  
   - Bayesiano INLA: `Rscript scripts/models/inla_spacetime.R`  
   - ConvLSTM: `python scripts/models/conv_lstm_train.py`

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

**Nota:** Los datos brutos no se incluyen en este repositorio por políticas de acceso. Los scripts de descarga y ejemplos sintéticos se proporcionan para reproducibilidad.

## 📈 Resultados esperados

- Comparación cuantitativa de modelos con mejora ≥15% en RMSE para eventos extremos.
- Modelo final (bayesiano o ConvLSTM) calibrado para la cuenca del río Chama.
- Propuesta operativa de umbrales de alerta temprana (percentiles dinámicos) con matriz de confusión.

## 🤝 Contribuciones

Este repositorio es parte de un trabajo de pregrado. No se aceptan contribuciones externas durante el desarrollo, pero se agradecen sugerencias vía issues.

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

## 📧 Contacto

- **Estudiante:** Sofía Mercado – sofiavmr20@gmail.com
- **Tutor:** Francisco Palm – fpalm@qu4nt.com  
- **Línea de investigación:** Estadística aplicada, modelado espacio-temporal, hidroclimatología

---

*Desarrollado en el marco del proyecto de grado para la Universided de Los Andes, Mérida, Venezuela.*
