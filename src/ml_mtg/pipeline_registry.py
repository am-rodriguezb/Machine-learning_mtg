"""Project pipelines."""

from kedro.pipeline import Pipeline
from ml_mtg.pipelines.data_prep import pipeline as dp
from ml_mtg.pipelines.classification import pipeline as cls
from ml_mtg.pipelines.regression import pipeline as reg

def register_pipelines() -> dict[str, Pipeline]:
    data_prep = dp.create_pipeline()
    # Pipelines independientes: classification y regression solo ejecutan sus propios nodos
    # Dependen de los datos preprocesados que ya fueron generados por data_prep
    classification_standalone = cls.create_pipeline()
    regression_standalone = reg.create_pipeline()
    # Pipelines completos para ejecución local (incluyen data_prep)
    classification_full = data_prep + cls.create_pipeline()
    regression_full = data_prep + reg.create_pipeline()
    return {
        "data_prep": data_prep,
        # Pipelines standalone para Airflow (asumen que data_prep ya se ejecutó)
        "classification": classification_standalone,
        "regression": regression_standalone,
        # Pipelines completos para ejecución local
        "classification_full": classification_full,
        "regression_full": regression_full,
        "__default__": data_prep
    }
