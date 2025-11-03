# 🎴 ML MTG - Análisis de Mazos de Magic: The Gathering

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Airflow](https://img.shields.io/badge/apache-airflow-2.9.0-orange)](https://airflow.apache.org/)
[![DVC](https://img.shields.io/badge/dvc-3.0+-blue)](https://dvc.org/)

Sistema de Machine Learning end-to-end para predecir la competitividad de mazos de Magic: The Gathering usando **Kedro**, **Airflow** y **DVC**.

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características Principales](#-características-principales)
- [Features de Ingeniería](#-features-de-ingeniería)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Orquestación con Airflow](#-orquestación-con-airflow)
- [Versionado con DVC](#-versionado-con-dvc)
- [Resultados](#-resultados)
- [Configuración](#-configuración)
- [Troubleshooting](#-troubleshooting)

## 🎯 Descripción

Este proyecto implementa un pipeline completo de Machine Learning para analizar y predecir la competitividad de mazos estándar de Magic: The Gathering. Utiliza técnicas avanzadas de feature engineering y múltiples algoritmos de ML para:

- **Clasificación Binaria**: Predecir si un mazo es competitivo (Tier 1-2) o no
- **Regresión**: Predecir el Competitiveness Score continuo (0.0-1.0)

### Pipeline de Procesamiento

```
Parsear Deck list → deck_cards
    ↓
Join con all_mtg_cards → merged_cards_decks
    ↓
Calcular Power Score y Mana Efficiency Score
    ↓
Derivar labels: competitivo (bin), competitiveness_score (reg)
    ↓
Entrenar ≥5 modelos por pipeline con GridSearchCV (cv≥5)
    ↓
Guardar outputs (CSV/JSON/plots) y versionar con DVC
    ↓
Orquestar en Airflow (features → clasificación & regresión)
```

## ✨ Características Principales

### 🔄 Pipelines Independientes
- **Pipeline de Clasificación**: Predice si un mazo es competitivo (Tier 1-2)
- **Pipeline de Regresión**: Predice el Competitiveness Score continuo
- **Pipeline de Preparación de Datos**: Feature engineering avanzado

### 🤖 Modelos Implementados

#### Clasificación (5 modelos):
- **LogisticRegression**: Regresión logística con regularización
- **SVC**: Support Vector Machine con kernels rbf y linear
- **RandomForestClassifier**: Ensemble de árboles de decisión
- **XGBClassifier**: Gradient boosting optimizado
- **KNeighborsClassifier**: K-Nearest Neighbors

#### Regresión (5 modelos):
- **LinearRegression**: Regresión lineal simple
- **Ridge**: Regresión con regularización L2
- **SVR**: Support Vector Regression
- **RandomForestRegressor**: Ensemble para regresión
- **XGBRegressor**: Gradient boosting para regresión

### 🎯 Búsqueda de Hiperparámetros
- **GridSearchCV** con validación cruzada (k=5 folds)
- Cross-validation estratificada para clasificación
- Cross-validation estándar para regresión
- Paralelismo limitado (n_jobs=2) para evitar OOM

### 📊 Métricas y Visualizaciones
- **Clasificación**: Accuracy, F1 Score (macro, weighted), ROC AUC, Precision, Recall
- **Regresión**: R², MAE, RMSE
- Tablas comparativas en HTML
- Gráficos interactivos con Plotly
- Gráficos estáticos con Matplotlib

## 🔬 Features de Ingeniería

El pipeline de preparación de datos genera **30+ features** avanzadas:

### 📈 Curva de Maná
- `avg_cmc`: Costo de maná promedio
- `std_cmc`: Desviación estándar del CMC
- `var_cmc`: Varianza del CMC

### 💎 Rareza y Composición
- `avg_rarity_weighted`: Rareza promedio ponderada por cantidad
- `total_cards`: Total de cartas en el mazo
- `pct_creatures`, `pct_instants`, `pct_sorceries`, `pct_planeswalkers`, `pct_enchantments`, `pct_artifacts`: Porcentajes de tipos de cartas

### 🔍 Keywords en Texto
- `pct_removal`: Porcentaje de cartas con capacidades de eliminación
- `pct_draw`: Porcentaje de cartas que permiten robar
- `pct_ramp`: Porcentaje de cartas de aceleración de maná
- `pct_counter`: Porcentaje de contadorespells
- `pct_lifegain`: Porcentaje de cartas que otorgan vida

### ⚔️ Cuerpo de Mesa (Board Presence)
- `avg_power`: Poder promedio de criaturas (ponderado por copias)
- `avg_toughness`: Resistencia promedio de criaturas (ponderado por copias)

### ⚡ Eficiencia de Maná
- `mana_efficiency`: Score de eficiencia (power+toughness)/cmc ponderado
- `power_score`: Power Score agregado del mazo (combinación de eficiencia, rareza y tipo)

### 🎨 Identidad de Color
- `avg_colors`: Promedio de colores por carta
- `pct_mono`: Porcentaje de cartas monocromáticas
- `pct_two_color`: Porcentaje de cartas bicolores
- `pct_three_plus`: Porcentaje de cartas con 3+ colores

### 🎴 Diversidad
- `unique_types`: Cantidad de tipos únicos de cartas en el mazo

## 📁 Estructura del Proyecto

```
kedro_mtg/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Configuración de datasets
│   │   ├── parameters.yml       # Parámetros de modelos
│   │   └── logging.yml
│   └── local/                   # Configuración local (no versionado)
├── data/
│   ├── 01_raw/                  # Datos crudos
│   │   ├── all_mtg_cards.csv
│   │   └── standard_decks.csv
│   ├── 02_intermediate/         # Datos intermedios
│   ├── 03_primary/              # Features finales y splits
│   ├── 06_models/               # Modelos entrenados (.pkl)
│   └── 08_reporting/            # Reportes y visualizaciones
├── src/
│   └── ml_mtg/
│       ├── pipelines/
│       │   ├── data_prep/       # Pipeline de preparación de datos
│       │   ├── classification/  # Pipeline de clasificación
│       │   └── regression/      # Pipeline de regresión
│       └── pipeline_registry.py
├── dags/
│   └── kedro_mtg_dag.py         # DAG de Airflow
├── dvc.yaml                     # Configuración DVC
├── docker-compose.yml           # Orquestación con Airflow
├── Dockerfile                   # Imagen Docker
├── requirements.txt             # Dependencias Python
└── pyproject.toml               # Configuración del proyecto
```

## 🚀 Instalación

### Prerrequisitos

- **Python** 3.9 o superior
- **Git**
- **Docker Desktop** (opcional, solo para Airflow)

### Instalación Local

1. **Clonar el repositorio**:
```bash
git clone https://github.com/TU_USUARIO/kedro_mtg.git
cd kedro_mtg
```

2. **Crear entorno virtual**:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Verificar instalación**:
```bash
kedro --version
```

## 💻 Uso

### Ejecutar Pipelines Localmente

#### Pipeline completo (preparación + ambos ML):
```bash
kedro run
```

#### Solo clasificación:
```bash
kedro run --pipeline=classification
```

#### Solo regresión:
```bash
kedro run --pipeline=regression
```

#### Solo preparación de datos:
```bash
kedro run --pipeline=data_prep
```

### Visualizar Resultados

Los resultados se generan automáticamente en `data/08_reporting/`:

- **`cls_results.csv`** - Tabla comparativa de modelos de clasificación
- **`reg_results.csv`** - Tabla comparativa de modelos de regresión
- **`cls_comparison_plot.png`** - Gráfico comparativo (Matplotlib)
- **`reg_comparison_plot.png`** - Gráfico comparativo (Matplotlib)
- **`cls_comparison_plotly.json`** - Gráfico interactivo (Plotly)
- **`reg_comparison_plotly.json`** - Gráfico interactivo (Plotly)
- **`cls_results.html`** / **`reg_results.html`** - Tablas HTML formateadas
- **`cls_metrics.json`** / **`reg_metrics.json`** - Métricas para DVC

### Ver Gráficos Interactivos

```bash
kedro viz
```

Luego abre http://localhost:4141 en tu navegador.

## ☁️ Orquestación con Airflow

### Requisitos

- **Docker Desktop** instalado y ejecutándose
- Al menos **8GB de RAM** disponible (configurado en docker-compose.yml)

### Iniciar Airflow

1. **Iniciar servicios**:
```bash
docker-compose up -d
```

2. **Acceder a la interfaz web**:
- URL: http://localhost:8080
- Usuario: `airflow`
- Contraseña: `airflow`

3. **Activar el DAG**:
- Busca el DAG `kedro_mtg_pipelines` en la interfaz
- Actívalo usando el toggle switch
- Ejecuta manualmente si lo deseas

### Estructura del DAG

```
data_preparation (prepara features)
    ↓
    ├─→ run_classification (ejecuta en paralelo)
    └─→ run_regression (ejecuta en paralelo)
```

### Configuración de Memoria

Los servicios de Airflow están configurados con:
- **8GB** de límite de memoria (`mem_limit: 8g`)
- **30 minutos** de timeout de heartbeat
- Optimizaciones para evitar OOM kills

## 📦 Versionado con DVC

### Inicializar DVC

```bash
# Si no está inicializado
dvc init

# Configurar almacenamiento remoto (opcional)
dvc remote add -d myremote s3://mybucket/path
```

### Usar DVC

```bash
# Reproducir todos los stages
dvc repro

# Reproducir un stage específico
dvc repro classification

# Ver métricas actuales
dvc metrics show

# Comparar métricas entre commits
dvc metrics diff

# Ver diferencias de plots
dvc plots diff
```

### Workflow Completo

```bash
# 1. Hacer cambios en código o parámetros
# 2. Reproducir pipeline
dvc repro

# 3. Revisar métricas
dvc metrics show

# 4. Commit cambios
git add .
git commit -m "feat: Mejora en features"
git push

# 5. Commit DVC (métricas y datos)
dvc commit
dvc push
```

## 📊 Resultados

### Modelos de Clasificación

Predicen si un mazo es **competitivo** (Tier 1-2) o no:

- **Target**: `competitive` (binario: 0 o 1)
- **Métricas**: Accuracy, F1 Score, ROC AUC
- **Mejor modelo**: Se guarda en `data/06_models/best_cls_model.pkl`

### Modelos de Regresión

Predicen el **Competitiveness Score** continuo (0.0-1.0):

- **Target**: `competitiveness_score` (continuo)
- **Métricas**: R², MAE, RMSE
- **Mejor modelo**: Se guarda en `data/06_models/best_reg_model.pkl`

## ⚙️ Configuración

### Parámetros de Modelos

Edita `conf/base/parameters.yml` para modificar:

- Hiperparámetros de búsqueda (`params_grid`)
- Número de folds para CV (`cv_folds`)
- Tamaño del test set (`test_size`)
- Random state (`random_state`)

### Ejemplo:

```yaml
cls_models:
  xgb:
    estimator: xgboost.XGBClassifier
    params_grid:
      n_estimators: [300, 500]
      max_depth: [4, 6]
      learning_rate: [0.05, 0.1]
```

### Datos de Entrada

Los datos deben estar en `data/01_raw/`:

- **`all_mtg_cards.csv`**: Base de datos completa de cartas MTG
  - Columnas requeridas: `name`, `cmc`, `rarity`, `color_identity`, `type`, `power`, `toughness`, `text`
- **`standard_decks.csv`**: Mazos estándar con sus Tiers
  - Columnas requeridas: `Name`, `Tier`, `Year`, `Deck list`

## 🐛 Troubleshooting

### Error: OOM Kill (exit code 137)

**Solución**: Ya está configurado con 8GB de memoria en docker-compose.yml. Si persiste:
- Aumenta `mem_limit` y `memswap_limit` en `docker-compose.yml`
- Reduce `n_jobs` en GridSearchCV (ya está en 2)
- Procesa datos en batches más pequeños

### Error: DVC no inicializado

```bash
dvc init
```

Asegúrate de que `.dvc/` esté en Git pero `.dvc/cache/` esté en `.gitignore`.

### Error: Datos no encontrados

Verifica que los archivos estén en `data/01_raw/`:
- `all_mtg_cards.csv`
- `standard_decks.csv`

### Error: Dependencias incompatibles

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Verificar Estado

```bash
# Ver pipelines disponibles
kedro pipeline list

# Ver configuración de catálogo
kedro catalog list

# Ejecutar con modo verbose
kedro run --verbose
```

## 📚 Tecnologías Utilizadas

- **[Kedro](https://kedro.org)**: Framework para pipelines de datos reproducibles
- **[Apache Airflow](https://airflow.apache.org/)**: Orquestación de workflows
- **[DVC](https://dvc.org/)**: Versionado de datos y experimentos
- **[scikit-learn](https://scikit-learn.org/)**: Machine Learning
- **[XGBoost](https://xgboost.ai/)**: Gradient boosting
- **[Pandas](https://pandas.pydata.org/)**: Manipulación de datos
- **[Plotly](https://plotly.com/)**: Visualizaciones interactivas
- **[Docker](https://www.docker.com/)**: Containerización

## 📝 Notas

- Los datos generados (`data/02_intermediate/`, `data/03_primary/`, etc.) están en `.gitignore` para mantener el repo ligero
- Usa DVC para versionar datos importantes
- Los modelos se guardan en formato Pickle (`.pkl`)
- Las métricas se exportan en JSON para compatibilidad con DVC

## 🔗 Recursos

- [Documentación Kedro](https://docs.kedro.org)
- [Documentación DVC](https://dvc.org/doc)
- [Documentación Airflow](https://airflow.apache.org/docs/)
- [Documentación scikit-learn](https://scikit-learn.org/stable/)

## 👤 Autor

**Amaro Rodriguez**

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!
