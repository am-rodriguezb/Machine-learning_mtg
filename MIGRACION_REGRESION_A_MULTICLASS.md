# Migración de Regresión a Clasificación Multiclase

## Resumen de Cambios

Se ha convertido el pipeline de **regresión** (predicción de winrate simulado) a **clasificación multiclase** (predicción de Tier 1-5).

### Objetivo Actualizado
- **Antes**: Predecir winrate continuo (0.0 - 1.0)
- **Ahora**: Predecir Tier (1-5) donde:
  - **Tier 1** = No optimizado / No competitivo
  - **Tier 2** = Poco optimizado
  - **Tier 3** = Moderadamente optimizado
  - **Tier 4** = Bien optimizado
  - **Tier 5** = Optimizado / Competitivo

## Archivos Modificados

### 1. `src/ml_mtg/pipelines/data_prep/nodes.py`
- ✅ Añadida variable objetivo `tier_multiclass` (Tier 1-5)
- ✅ Actualizada función `split_train_test` para generar splits multiclase
- ✅ Eliminada variable `Winrate` simulada

### 2. `src/ml_mtg/pipelines/regression/nodes.py`
- ✅ Convertido `train_regression_models` → `train_multiclass_models`
- ✅ Métricas cambiadas: R²/MAE/RMSE → Accuracy/F1/Precision/Recall
- ✅ Actualizado `create_reg_comparison_plots` → `create_multiclass_comparison_plots`
- ✅ Actualizado `save_reg_metrics_table` → `save_multiclass_metrics_table`

### 3. `src/ml_mtg/pipelines/regression/pipeline.py`
- ✅ Actualizado para usar funciones multiclase
- ✅ Inputs cambiados de `X_train_reg/y_train_reg` → `X_train_multiclass/y_train_multiclass`
- ✅ Outputs actualizados con prefijo `multiclass_*`

### 4. `conf/base/parameters.yml`
- ✅ Reemplazado `reg_models` → `multiclass_models`
- ✅ Modelos actualizados para clasificación multiclase:
  - `logreg_multiclass`: LogisticRegression con multi_class
  - `svm_multiclass`: SVC
  - `rf_multiclass`: RandomForestClassifier
  - `xgb_multiclass`: XGBClassifier
  - `knn_multiclass`: KNeighborsClassifier

### 5. `conf/base/catalog.yml`
- ✅ Actualizados datasets de `*_reg` → `*_multiclass`
- ✅ Nuevos outputs:
  - `multiclass_model_results`
  - `multiclass_metrics_json`
  - `multiclass_comparison_plot`
  - `multiclass_comparison_plotly`
  - `multiclass_results_html`
  - `best_multiclass_model`

### 6. `src/ml_mtg/pipeline_registry.py`
- ✅ Pipeline renombrado: `regression` → `multiclass`
- ✅ Mantiene compatibilidad con pipelines standalone y full

### 7. `dvc.yaml`
- ✅ Stage renombrado: `regression` → `multiclass`
- ✅ Dependencias y outputs actualizados

### 8. `dags/kedro_mtg_dag.py`
- ✅ Task renombrado: `run_regression` → `run_multiclass`
- ✅ Descripción y tags actualizados

## Nuevos Datasets Generados

El pipeline `data_prep` ahora genera:
- `X_train_multiclass.parquet` / `X_test_multiclass.parquet`
- `y_train_multiclass.parquet` / `y_test_multiclass.parquet`

## Nuevos Outputs del Pipeline Multiclase

- `data/06_models/best_multiclass_model.pkl`
- `data/08_reporting/multiclass_results.csv`
- `data/08_reporting/multiclass_comparison_plot.png`
- `data/08_reporting/multiclass_comparison_plotly.json`
- `data/08_reporting/multiclass_results.html`
- `data/08_reporting/multiclass_model_predictions.pkl`
- `data/08_reporting/multiclass_metrics.json` (métrica para DVC)

## Uso de DVC

Para ejecutar con DVC:

```bash
# Ejecutar todos los stages
dvc repro

# Ejecutar solo un stage específico
dvc repro multiclass

# Ver métricas
dvc metrics show

# Ver diferencias de métricas (si hay commits previos)
dvc metrics diff
```

## Nota Importante

Los archivos antiguos de regresión (`*_reg*`) ya no se generan. Si necesitas mantener compatibilidad, puedes crear un alias o pipeline adicional.

## Verificación

Para verificar que todo funciona:

```bash
# Ver pipelines disponibles
kedro pipeline list

# Ejecutar pipeline multiclase
kedro run --pipeline=multiclass

# Ejecutar desde DVC
dvc repro multiclass
```

