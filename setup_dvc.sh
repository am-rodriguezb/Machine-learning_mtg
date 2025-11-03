#!/bin/bash
# Script de configuración inicial de DVC para el proyecto kedro_mtg

echo "=== Configuración de DVC para kedro_mtg ==="

# Verificar si DVC está instalado
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC no está instalado. Instalando..."
    pip install dvc dvc-s3
    echo "✅ DVC instalado"
else
    echo "✅ DVC ya está instalado"
fi

# Inicializar DVC si no está inicializado
if [ ! -d ".dvc" ]; then
    echo "📦 Inicializando DVC..."
    dvc init
    echo "✅ DVC inicializado"
else
    echo "✅ DVC ya está inicializado"
fi

# Configurar remoto (opcional - comenta si no tienes remoto)
# Ejemplo con Google Drive:
# echo "📤 Configurando remoto DVC (Google Drive)..."
# dvc remote add -d storage gdrive://tu-folder-id
# echo "✅ Remoto configurado"

# Ejemplo con S3 (descomenta y configura):
# echo "📤 Configurando remoto DVC (S3)..."
# dvc remote add -d storage s3://mi-bucket/kedro-mtg
# dvc remote modify storage credentialpath ~/.aws/credentials
# echo "✅ Remoto S3 configurado"

# Versionar datasets grandes (opcional - solo si los quieres trackear con DVC)
# echo "📊 Versionando datasets grandes..."
# dvc add data/01_raw/all_mtg_cards.csv
# dvc add data/01_raw/standard_decks.csv
# git add data/01_raw/*.csv.dvc data/01_raw/.gitignore
# git commit -m "Track datasets grandes con DVC" || echo "⚠️  Git commit omitido (repositorio no inicializado o sin cambios)"

echo ""
echo "=== Verificación ==="
echo "📋 Verificando configuración DVC..."
dvc version
echo ""
echo "📋 Verificando pipeline DVC..."
dvc dag
echo ""
echo "✅ Configuración DVC completada"
echo ""
echo "📝 Próximos pasos:"
echo "1. Configura un remoto (S3, GDrive, etc.) si lo deseas:"
echo "   dvc remote add -d storage <url-remoto>"
echo ""
echo "2. Ejecuta el pipeline completo:"
echo "   dvc repro"
echo ""
echo "3. Ver métricas:"
echo "   dvc metrics show"
echo ""
echo "4. Sube datos al remoto (si configuraste uno):"
echo "   dvc push"

