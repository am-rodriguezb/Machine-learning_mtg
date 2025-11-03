# Cómo Activar el Entorno Virtual

## Situación Actual
Tu entorno virtual está en `.venv/venv/` (anidado). 

## Solución Rápida (Activar el venv actual)

### En PowerShell:
```powershell
.\.venv\venv\Scripts\Activate.ps1
```

### En CMD:
```cmd
.venv\venv\Scripts\activate.bat
```

### En Git Bash:
```bash
source .venv/venv/Scripts/activate
```

## Solución Recomendada (Recrear el venv correctamente)

1. **Eliminar el venv actual**:
```powershell
Remove-Item -Recurse -Force .venv
```

2. **Crear nuevo venv**:
```powershell
python -m venv .venv
```

3. **Activar**:
```powershell
.\.venv\Scripts\Activate.ps1
```

4. **Instalar dependencias**:
```powershell
pip install -r requirements.txt
pip install -e .
```

## Verificar que está activado

Cuando el venv está activado, verás `(.venv)` o `(venv)` al inicio de la línea de comandos:

```
(.venv) PS C:\Users\xxama\Proyectos\kedro_mtg>
```

## Desactivar

Simplemente ejecuta:
```powershell
deactivate
```

