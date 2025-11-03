import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def _build_estimator(path: str):
    module_name, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()

def train_regression_models(X_train, y_train, X_test, y_test, reg_models: dict, cv_folds: int):
    """
    Entrena múltiples modelos de REGRESIÓN (Competitiveness Score continuo) con GridSearchCV.
    Modelos: LinearRegression, Ridge, SVR, RandomForestRegressor, XGBRegressor.
    Retorna resultados comparativos y el mejor modelo.
    """
    # Asegurar que y_train y y_test son Series de numpy
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    
    rows = []
    best_model = None
    best_score = -np.inf
    model_predictions = {}

    for name, spec in reg_models.items():
        estimator = _build_estimator(spec['estimator'])
        # Pipeline con scaler para modelos sensibles a escala
        need_scale = any(k in spec['estimator'] for k in ['LinearRegression','Ridge','SVR'])
        if need_scale:
            est = SkPipeline([('scaler', StandardScaler()), ('model', estimator)])
            # Prefijar grid con 'model__'
            grid = {f"model__{k}": v for k, v in spec.get('params_grid', {}).items()}
        else:
            est = estimator
            grid = spec.get('params_grid', {})

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        # Reducir n_jobs a 2 para evitar OOM kills
        gs = GridSearchCV(est, grid, cv=cv, scoring='r2', n_jobs=2, verbose=0)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        rows.append([name, r2, mae, rmse, gs.best_params_])
        model_predictions[name] = {'y_pred': y_pred, 'y_test': y_test}

        if r2 > best_score:
            best_score = r2
            best_model = gs.best_estimator_

    results = pd.DataFrame(rows, columns=['model','r2','mae','rmse','best_params'])
    return results, best_model, model_predictions

def create_reg_comparison_plots(results: pd.DataFrame, model_predictions: dict):
    """
    Crea visualizaciones comparativas de modelos de regresión.
    Retorna figuras de matplotlib y plotly.
    """
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # 1. Gráfico de barras comparativo (matplotlib)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Métricas por modelo
    metrics_df = results[['model', 'r2', 'mae', 'rmse']].copy()
    
    # R²
    axes[0, 0].barh(metrics_df['model'], metrics_df['r2'], color='lightblue')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_title('R² Score por Modelo (mayor es mejor)')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Baseline (R²=0)')
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].barh(metrics_df['model'], metrics_df['mae'], color='lightcoral')
    axes[0, 1].set_xlabel('MAE (Mean Absolute Error)')
    axes[0, 1].set_title('MAE por Modelo (menor es mejor)')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # RMSE
    axes[1, 0].barh(metrics_df['model'], metrics_df['rmse'], color='lightgreen')
    axes[1, 0].set_xlabel('RMSE (Root Mean Squared Error)')
    axes[1, 0].set_title('RMSE por Modelo (menor es mejor)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Scatter plot: Predicted vs Actual (para el mejor modelo)
    best_model_name = results.sort_values('r2', ascending=False).iloc[0]['model']
    best_preds = model_predictions[best_model_name]
    axes[1, 1].scatter(best_preds['y_test'], best_preds['y_pred'], alpha=0.5)
    # Línea ideal (y = x)
    min_val = min(best_preds['y_test'].min(), best_preds['y_pred'].min())
    max_val = max(best_preds['y_test'].max(), best_preds['y_pred'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    axes[1, 1].set_xlabel('Actual Competitiveness Score')
    axes[1, 1].set_ylabel('Predicted Competitiveness Score')
    axes[1, 1].set_title(f'Predicted vs Actual - {best_model_name}')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 2. Tabla HTML con resultados (como string)
    results_html = results.to_html(classes='table table-striped', index=False, escape=False)
    
    # 3. Gráfico interactivo Plotly
    fig_plotly = go.Figure()
    
    fig_plotly.add_trace(go.Bar(
        name='R² Score',
        x=results['model'],
        y=results['r2'],
        marker_color='lightblue',
        text=[f'{r:.3f}' for r in results['r2']],
        textposition='auto'
    ))
    
    fig_plotly.update_layout(
        title='Comparación de Modelos de Regresión (Competitiveness Score)',
        xaxis_title='Modelo',
        yaxis_title='R² Score',
        barmode='group',
        height=600,
        template='plotly_white'
    )
    
    # Agregar scatter plot interactivo para el mejor modelo
    fig_scatter = go.Figure()
    best_preds = model_predictions[best_model_name]
    fig_scatter.add_trace(go.Scatter(
        x=best_preds['y_test'],
        y=best_preds['y_pred'],
        mode='markers',
        marker=dict(color='lightblue', opacity=0.6),
        name='Predictions'
    ))
    # Línea ideal
    min_val = min(float(best_preds['y_test'].min()), float(best_preds['y_pred'].min()))
    max_val = max(float(best_preds['y_test'].max()), float(best_preds['y_pred'].max()))
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Ideal (y=x)'
    ))
    fig_scatter.update_layout(
        title=f'Predicted vs Actual - {best_model_name}',
        xaxis_title='Actual Competitiveness Score',
        yaxis_title='Predicted Competitiveness Score',
        height=600,
        template='plotly_white'
    )
    
    return fig, fig_plotly, fig_scatter, results_html

def save_reg_metrics_table(results: pd.DataFrame) -> dict:
    """Guarda métricas en formato JSON para DVC."""
    best = results.sort_values('r2', ascending=False).iloc[0]
    return {
        "best_model": best['model'],
        "best_r2": float(best['r2']),
        "best_mae": float(best['mae']),
        "best_rmse": float(best['rmse']),
        "all_models": results.to_dict('records')
    }
