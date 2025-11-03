"""
Inicializador de Kedro para notebooks.
Carga el contexto, catálogo y pipelines automáticamente.
Compatible con Kedro 1.0.0.
"""

import sys, os

# 🔧 Asegura que src esté en el path antes de importar ml_mtg
project_root = os.path.abspath("..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from ml_mtg.pipeline_registry import register_pipelines

def init_kedro():
    # Subir un nivel desde /notebooks hasta la raíz del proyecto
    project_root = os.path.abspath("..")
    src_path = os.path.join(project_root, "src/")

    if src_path not in sys.path:
        sys.path.append(src_path)

    configure_project("ml_mtg")

    session = KedroSession.create(project_path=project_root)
    context = session.load_context()

    catalog = context.catalog
    pipelines = register_pipelines()
    print("🚀 Kedro cargado correctamente")

    # 兼容 Kedro 1.0: buscar el catálogo interno real
    dataset_names = []
    try:
        # estructura típica: catalog._catalog.catalog._data_sets
        inner_catalog = getattr(catalog, "_catalog", None)
        if inner_catalog and hasattr(inner_catalog, "catalog"):
            dataset_names = list(inner_catalog.catalog._data_sets.keys())
        elif inner_catalog and hasattr(inner_catalog, "_data_sets"):
            dataset_names = list(inner_catalog._data_sets.keys())
    except Exception as e:
        print("⚠️ No se pudieron listar datasets:", e)

    print("📦 Datasets disponibles:", dataset_names)
    print("🧩 Pipelines registrados:", list(pipelines.keys()))

    return catalog, pipelines