# Proyecto Kedro MTG - Guía de Configuración

Este proyecto implementa dos pipelines independientes de Machine Learning (clasificación y regresión) para análisis de mazos de Magic: The Gathering.

## ✅ Características Implementadas

### 1. Pipelines Independientes
- **Pipeline de Clasificación**: Predice si un mazo es competitivo (Tier 1-2) o no
- **Pipeline de Regresión**: Predice el winrate de un mazo

### 2. Modelos Implementados

#### Clasificación (5 modelos):
- Logistic Regression
- SVM (Support Vector Machine)
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)

#### Regresión (7 modelos):
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- SVR (Support Vector Regression)
- K-Nearest Neighbors Regressor

### 3. Búsqueda de Hiperparámetros
- GridSearchCV con validación cruzada (k=5 folds)
- Cross-validation estratificada para clasificación
- Cross-validation estándar para regresión

### 4. Métricas y Visualizaciones
- **Clasificación**: Accuracy, F1 Score, ROC AUC
- **Regresión**: R², MAE, RMSE
- Tablas comparativas en HTML
- Gráficos interactivos con Plotly
- Gráficos estáticos con Matplotlib

### 5. Orquestación con Airflow
- DAG configurado para ejecutar ambos pipelines
- Dependencias correctas entre tasks
- Ejecución programada diaria

### 6. Versionado con DVC
- Datasets versionados
- Features versionadas
- Modelos versionados
- Métricas versionadas

### 7. Docker
- Dockerfile para ejecución reproducible
- docker-compose.yml con Airflow completo

## 📁 Estructura del Proyecto

```
kedro_mtg/
├── conf/
│   └── base/
│       ├── catalog.yml          # Configuración de datasets
│       ├── parameters.yml       # Parámetros de modelos
│       └── logging.yml
├── data/
│   ├── 01_raw/                  # Datos crudos
│   ├── 02_intermediate/         # Datos intermedios
│   ├── 03_primary/              # Features finales
│   ├── 06_models/               # Modelos entrenados
│   └── 08_reporting/            # Reportes y visualizaciones
├── src/
│   └── ml_mtg/
│       ├── pipelines/
│       │   ├── data_prep/       # Pipeline de preparación
│       │   ├── classification/  # Pipeline de clasificación
│       │   └── regression/      # Pipeline de regresión
│       └── pipeline_registry.py
├── dags/
│   └── kedro_mtg_dag.py         # DAG de Airflow
├── dvc.yaml                     # Configuración DVC
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml           # Orquestación completa
└── requirements.txt             # Dependencias
```

## 🚀 Instalación y Uso

### Instalación Local

1. **Crear entorno virtual**:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Ejecutar pipelines**:
```bash
# Pipeline completo (data prep + ambos ML)
kedro run

# Solo clasificación
kedro run --pipeline=classification

# Solo regresión
kedro run --pipeline=regression
```

### Usando Docker

1. **Construir imagen**:
```bash
docker build -t kedro-mtg .
```

2. **Ejecutar contenedor**:
```bash
docker run -v $(pwd)/data:/app/data kedro-mtg
```

### Usando Docker Compose (con Airflow)

1. **Iniciar servicios**:
```bash
docker-compose up -d
```

2. **Acceder a Airflow**:
- Abre http://localhost:8080
- Usuario: `airflow`
- Contraseña: `airflow`

3. **Ver resultados**:
Los resultados se generan en `data/08_reporting/`:
- `cls_results.csv` - Tabla de resultados clasificación
- `reg_results.csv` - Tabla de resultados regresión
- `cls_comparison_plot.png` - Gráfico comparativo clasificación
- `reg_comparison_plot.png` - Gráfico comparativo regresión
- `cls_comparison_plotly.json` - Gráfico interactivo clasificación
- `reg_comparison_plotly.json` - Gráfico interactivo regresión

### Usando DVC

1. **Inicializar DVC** (si no está inicializado):
```bash
dvc init
```

2. **Reproducir pipeline**:
```bash
dvc repro
```

3. **Ver métricas**:
```bash
dvc metrics show
```

4. **Comparar versiones**:
```bash
dvc metrics diff
```

## 📊 Visualización de Resultados

Los gráficos se generan automáticamente en `data/08_reporting/`:

- **Matplotlib**: Gráficos estáticos comparativos
- **Plotly**: Gráficos interactivos (abrir con `kedro viz`)
- **HTML**: Tablas HTML formateadas

Para ver los gráficos interactivos:
```bash
kedro viz
```
Luego abrir http://localhost:4141

## 🔧 Configuración de Modelos

Los modelos y sus hiperparámetros se configuran en `conf/base/parameters.yml`:

```yaml
cls_models:
  logreg:
    estimator: sklearn.linear_model.LogisticRegression
    params_grid:
      C: [0.1, 1, 10]
      # ... más parámetros
```

Puedes agregar más modelos modificando este archivo.

## 📝 Notas

- Los datos de entrada deben estar en `data/01_raw/`:
  - `all_mtg_cards.csv`
  - `standard_decks.csv`
- El pipeline de data prep crea las variables objetivo automáticamente
- Los modelos se guardan en `data/06_models/`
- Las métricas se guardan en formato JSON para DVC en `data/08_reporting/`

## 🐛 Troubleshooting

Si encuentras errores:

1. **Verificar que los datos estén en `data/01_raw/`**
2. **Verificar que las dependencias estén instaladas**: `pip install -r requirements.txt`
3. **Verificar que el entorno virtual esté activado**
4. **Ejecutar con modo verbose**: `kedro run --verbose`

## 📚 Recursos

- [Documentación Kedro](https://docs.kedro.org)
- [Documentación DVC](https://dvc.org/doc)
- [Documentación Airflow](https://airflow.apache.org/docs/)

