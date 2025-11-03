from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_regression_models, save_reg_metrics_table, create_reg_comparison_plots

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_regression_models,
            inputs=["X_train_reg","y_train_reg","X_test_reg","y_test_reg","params:reg_models","params:cv_folds"],
            outputs=["reg_model_results","best_reg_model","reg_model_predictions"],
            name="train_regression_models"
        ),
        node(
            func=create_reg_comparison_plots,
            inputs=["reg_model_results","reg_model_predictions"],
            outputs=["reg_comparison_plot","reg_comparison_plotly","reg_scatter_plotly","reg_results_html"],
            name="create_reg_plots"
        ),
        node(
            func=save_reg_metrics_table,
            inputs="reg_model_results",
            outputs="reg_metrics_json",
            name="save_reg_metrics"
        )
    ])
