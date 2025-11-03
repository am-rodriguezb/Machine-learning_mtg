import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def _build_estimator(path: str):
    # path tipo "sklearn.linear_model.LogisticRegression"
    module_name, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()

def train_classification_models(X_train, y_train, X_test, y_test, cls_models: dict, cv_folds: int):
    """
    Entrena múltiples modelos de clasificación con GridSearchCV y cross-validation.
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

    for name, spec in cls_models.items():
        estimator = _build_estimator(spec['estimator'])
        # pipeline con scaler para modelos sensibles a escala
        need_scale = any(k in spec['estimator'] for k in ['LogisticRegression','SVC','KNeighbors'])
        if need_scale:
            est = SkPipeline([('scaler', StandardScaler()), ('model', estimator)])
            # prefijar grid con 'model__'
            grid = {f"model__{k}": v for k, v in spec.get('params_grid', {}).items()}
        else:
            est = estimator
            grid = spec.get('params_grid', {})

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        # Reducir n_jobs a 2 para evitar OOM kills (usa menos memoria que n_jobs=-1)
        gs = GridSearchCV(est, grid, cv=cv, scoring='f1', n_jobs=2, verbose=0)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        try:
            y_proba = gs.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        rows.append([name, acc, f1, auc, gs.best_params_])
        model_predictions[name] = {'y_pred': y_pred, 'y_test': y_test}

        if f1 > best_score:
            best_score = f1
            best_model = gs.best_estimator_

    results = pd.DataFrame(rows, columns=['model','accuracy','f1','roc_auc','best_params'])
    return results, best_model, model_predictions

def create_cls_comparison_plots(results: pd.DataFrame, model_predictions: dict):
    """
    Crea visualizaciones comparativas de modelos de clasificación.
    Retorna figuras de matplotlib y plotly.
    """
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # 1. Gráfico de barras comparativo (matplotlib)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Métricas por modelo
    metrics_df = results[['model', 'accuracy', 'f1', 'roc_auc']].copy()
    metrics_df['roc_auc'] = metrics_df['roc_auc'].fillna(0)
    
    # Accuracy
    axes[0, 0].barh(metrics_df['model'], metrics_df['accuracy'], color='skyblue')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Accuracy por Modelo')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # F1 Score
    axes[0, 1].barh(metrics_df['model'], metrics_df['f1'], color='lightgreen')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('F1 Score por Modelo')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # ROC AUC
    axes[1, 0].barh(metrics_df['model'], metrics_df['roc_auc'], color='salmon')
    axes[1, 0].set_xlabel('ROC AUC')
    axes[1, 0].set_title('ROC AUC por Modelo')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Comparación múltiple
    metrics_long = metrics_df.melt(id_vars='model', var_name='metric', value_name='score')
    sns.barplot(data=metrics_long, x='score', y='model', hue='metric', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_title('Comparación de Métricas')
    axes[1, 1].legend(title='Métrica')
    
    plt.tight_layout()
    
    # 2. Tabla HTML con resultados (como string)
    results_html = results.to_html(classes='table table-striped', index=False, escape=False)
    
    # 3. Gráfico interactivo Plotly
    fig_plotly = go.Figure()
    
    fig_plotly.add_trace(go.Bar(
        name='Accuracy',
        x=results['model'],
        y=results['accuracy'],
        marker_color='lightblue'
    ))
    fig_plotly.add_trace(go.Bar(
        name='F1 Score',
        x=results['model'],
        y=results['f1'],
        marker_color='lightgreen'
    ))
    fig_plotly.add_trace(go.Bar(
        name='ROC AUC',
        x=results['model'],
        y=results['roc_auc'].fillna(0),
        marker_color='salmon'
    ))
    
    fig_plotly.update_layout(
        title='Comparación de Modelos de Clasificación',
        xaxis_title='Modelo',
        yaxis_title='Score',
        barmode='group',
        height=600,
        template='plotly_white'
    )
    
    return fig, fig_plotly, results_html

def save_cls_metrics_table(results: pd.DataFrame) -> dict:
    """Guarda métricas en formato JSON para DVC."""
    best = results.sort_values('f1', ascending=False).iloc[0]
    return {
        "best_model": best['model'],
        "best_f1": float(best['f1']),
        "best_accuracy": float(best['accuracy']),
        "best_auc": float(best['roc_auc']) if not pd.isna(best['roc_auc']) else None,
        "all_models": results.to_dict('records')
    }
