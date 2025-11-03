from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_classification_models, save_cls_metrics_table, create_cls_comparison_plots

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_classification_models,
            inputs=["X_train_cls","y_train_cls","X_test_cls","y_test_cls","params:cls_models","params:cv_folds"],
            outputs=["cls_model_results","best_cls_model","cls_model_predictions"],
            name="train_cls_models"
        ),
        node(
            func=create_cls_comparison_plots,
            inputs=["cls_model_results","cls_model_predictions"],
            outputs=["cls_comparison_plot","cls_comparison_plotly","cls_results_html"],
            name="create_cls_plots"
        ),
        node(
            func=save_cls_metrics_table,
            inputs="cls_model_results",
            outputs="cls_metrics_json",
            name="save_cls_metrics"
        )
    ])
