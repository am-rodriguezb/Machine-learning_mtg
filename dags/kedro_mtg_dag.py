from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Definir el DAG
with DAG(
    dag_id="kedro_mtg_pipelines",
    default_args=default_args,
    description="DAG para ejecutar pipelines de clasificación binaria y regresión (Competitiveness Score) de MTG",
    schedule_interval="@daily",  # Ejecutar diariamente
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["kedro", "ml", "mtg", "classification", "regression"],
) as dag:

    # Función para preparar el comando con instalación de dependencias
    def get_kedro_command(pipeline_name):
        return f"""
        set -e &&
        echo "=== Iniciando instalación y ejecución del pipeline {pipeline_name} ===" &&
        echo "Directorio de trabajo inicial: $(pwd)" &&
        PROJ_ROOT="/opt/airflow" &&
        cd "$PROJ_ROOT" &&
        echo "Directorio raíz del proyecto: $PROJ_ROOT" &&
        echo "Buscando pyproject.toml..." &&
        if [ -f "/opt/airflow/pyproject.toml" ]; then
          echo "✓ pyproject.toml encontrado en /opt/airflow/" &&
          cp /opt/airflow/pyproject.toml "$PROJ_ROOT/pyproject.toml" 2>/dev/null || true
        elif [ -f "/opt/airflow/kedro_project/pyproject.toml" ]; then
          echo "✓ pyproject.toml encontrado en kedro_project, copiando..." &&
          cp /opt/airflow/kedro_project/pyproject.toml "$PROJ_ROOT/pyproject.toml"
        elif [ -f "/opt/airflow/dags/../pyproject.toml" ]; then
          echo "✓ pyproject.toml encontrado, copiando..." &&
          cp /opt/airflow/dags/../pyproject.toml "$PROJ_ROOT/pyproject.toml"
        else
          echo "✗ pyproject.toml no encontrado en ubicaciones esperadas" &&
          echo "Buscando en todo el sistema..." &&
          find /opt/airflow -name "pyproject.toml" -type f 2>/dev/null | head -5 &&
          PYPROJ_FOUND=$(find /opt/airflow -name "pyproject.toml" -type f 2>/dev/null | head -1) &&
          if [ -n "$PYPROJ_FOUND" ] && [ -f "$PYPROJ_FOUND" ]; then
            echo "✓ pyproject.toml encontrado en: $PYPROJ_FOUND, copiando..." &&
            cp "$PYPROJ_FOUND" "$PROJ_ROOT/pyproject.toml"
          else
            echo "⚠ pyproject.toml no encontrado, creando uno con configuración mínima..." &&
            python3 << 'PYEOF'
import os
proj_root = "/opt/airflow"
pyproject_content = '''[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
requires-python = ">=3.9"
name = "ml_mtg"
version = "0.1.0"
dependencies = []

[tool.kedro]
package_name = "ml_mtg"
project_name = "ml_mtg"
kedro_init_version = "1.0.0"
source_dir = "src"

[tool.setuptools.packages.find]
where = ["src"]
namespaces = false
'''
with open(os.path.join(proj_root, "pyproject.toml"), "w") as f:
    f.write(pyproject_content)
print("pyproject.toml creado")
PYEOF
          fi
        fi &&
        echo "Verificando archivos..." &&
        ls -la "$PROJ_ROOT/pyproject.toml" "$PROJ_ROOT/requirements.txt" 2>/dev/null || echo "Algunos archivos faltan" &&
        if [ ! -f "$PROJ_ROOT/pyproject.toml" ]; then
          echo "✗ ERROR: pyproject.toml no disponible" &&
          exit 1
        fi &&
        if [ ! -f "$PROJ_ROOT/requirements.txt" ]; then
          echo "✗ ERROR: requirements.txt no encontrado" &&
          exit 1
        fi &&
        echo "✓ pyproject.toml encontrado/creado" &&
        echo "✓ requirements.txt encontrado" &&
        export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH &&
        echo "Instalando dependencias..." &&
        pip install --quiet --no-cache-dir -r "$PROJ_ROOT/requirements.txt" &&
        echo "Instalando proyecto en modo desarrollo..." &&
        pip install --quiet --no-cache-dir -e "$PROJ_ROOT" &&
        echo "Verificando instalación de kedro..." &&
        kedro --version &&
        cd "$PROJ_ROOT" &&
        echo "Ejecutando pipeline {pipeline_name}..." &&
        kedro run --pipeline={pipeline_name} &&
        echo "=== Pipeline {pipeline_name} completado exitosamente ==="
        """

    # Task para preparar datos (requisito para ambos pipelines)
    data_prep = BashOperator(
        task_id="data_preparation",
        bash_command=get_kedro_command("data_prep"),
    )

    # Task para pipeline de clasificación
    classification = BashOperator(
        task_id="run_classification",
        bash_command=get_kedro_command("classification"),
    )

    # Task para pipeline de regresión (Competitiveness Score continuo)
    regression = BashOperator(
        task_id="run_regression",
        bash_command=get_kedro_command("regression"),
    )

    # Definir dependencias: ambos pipelines dependen de data_prep
    # y pueden ejecutarse en paralelo después de data_prep
    data_prep >> [classification, regression]
