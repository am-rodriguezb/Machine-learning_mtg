# Carpeta .dvc - Explicación y Configuración

## ¿Para qué sirve la carpeta `.dvc`?

La carpeta `.dvc` es el **núcleo de configuración de DVC**. Contiene:

1. **Metadatos de configuración** (`config`): Configuración del repositorio DVC
2. **Cache local** (`cache/`): Copias locales de los datos versionados (NO se sube a Git)
3. **Archivos temporales** (`tmp/`): Locks y archivos temporales
4. **`.gitignore`**: Indica qué archivos de `.dvc/` NO trackear (solo cache y tmp)

## Estructura de la carpeta `.dvc/`

```
.dvc/
├── config          # ⭐ Configuración principal (se trackea en Git)
├── config.local    # ⚠️ Configuración local/privada (NO se trackea)
├── .gitignore      # Qué ignorar dentro de .dvc/
├── cache/          # Cache local de datos (NO se trackea)
│   ├── files/      # Archivos cacheados
│   └── runs/       # Historial de ejecuciones
└── tmp/            # Archivos temporales (NO se trackea)
```

## Archivo `.dvc/config`

### Estado actual (vacío = configuración por defecto)

Un archivo `config` vacío significa que DVC está usando:
- **Cache local** únicamente
- **Sin remoto configurado** (todos los datos quedan en tu máquina)

### Configuración básica (actual - sin remoto)

```ini
[core]
    remote = 

[cache]
    type = reflink,copy
    dir = .dvc/cache
```

Esta configuración **es suficiente** si:
- ✅ Trabajas localmente
- ✅ No necesitas compartir datos con otros
- ✅ No necesitas backups en la nube

### Configuración con remoto (opcional)

Si quieres usar **Google Drive**, **S3**, **Azure**, etc.:

#### Ejemplo: Google Drive
```ini
['remote "storage"']
    url = gdrive://tu-folder-id
    description = Almacenamiento remoto para datasets grandes

[core]
    remote = storage
```

#### Ejemplo: Amazon S3
```ini
['remote "storage"']
    url = s3://mi-bucket/kedro-mtg
    credentialpath = ~/.aws/credentials
    endpointurl = https://s3.amazonaws.com

[core]
    remote = storage
```

#### Ejemplo: Azure Blob Storage
```ini
['remote "storage"']
    url = azure://mi-container/kedro-mtg
    connection_string = DefaultEndpointsProtocol=https;AccountName=...

[core]
    remote = storage
```

## Archivo `.dvc/.gitignore`

Este archivo indica qué NO trackear dentro de `.dvc/`:
```
/config.local    # Configuraciones locales/privadas
/tmp            # Archivos temporales
/cache          # Cache de datos (muy pesado para Git)
```

**IMPORTANTE**: El archivo `config` SÍ se debe trackear (por eso no está en .gitignore)

## Archivo `.dvcignore`

Este archivo está en la **raíz del proyecto** (no dentro de `.dvc/`) y le dice a DVC qué archivos **no debe versionar**:

```
# Archivos que DVC debe ignorar (mejora performance)
venv/
__pycache__/
*.pyc
.git/
```

## ¿Qué trackear en Git vs qué no?

### ✅ SÍ trackear en Git:
- `.dvc/config` - Configuración compartida
- `.dvc/.gitignore` - Qué ignorar dentro de .dvc
- `dvc.yaml` - Definición de pipelines
- `dvc.lock` - Versiones exactas de datos/modelos

### ❌ NO trackear en Git:
- `.dvc/cache/` - Cache local (muy pesado)
- `.dvc/tmp/` - Temporales
- `.dvc/config.local` - Credenciales locales

## Flujo de trabajo típico

1. **Sin remoto** (tu caso actual):
   ```bash
   dvc repro          # Ejecuta pipeline
   dvc metrics show   # Ver métricas
   ```

2. **Con remoto** (si configuras uno):
   ```bash
   dvc repro          # Ejecuta pipeline
   dvc push           # Sube datos al remoto
   dvc pull           # Descarga datos del remoto
   ```

## Resumen

- **`.dvc/config`**: Configuración del repositorio (remotos, cache, etc.)
- **`.dvc/cache/`**: Datos cacheados localmente (NO subir a Git)
- **`.dvc/.gitignore`**: Qué ignorar dentro de .dvc/
- **`.dvcignore`**: Qué archivos DVC debe ignorar en todo el proyecto

**Para tu proyecto actual**: El `config` vacío está bien si trabajas localmente sin remoto.

