# Instalar Docker Desktop en Windows

## Paso 1: Descargar Docker Desktop

1. Ve a: https://www.docker.com/products/docker-desktop/
2. Descarga **Docker Desktop for Windows**
3. Ejecuta el instalador `.exe` descargado

## Paso 2: Requisitos del Sistema

Docker Desktop requiere:
- Windows 10 64-bit: Pro, Enterprise, o Education (Build 15063 o superior)
- Windows 11 64-bit: Home o Pro versión 21H2 o superior
- Habilitar **WSL 2** (Windows Subsystem for Linux 2)

### Habilitar WSL 2 (si es necesario)

Si tu sistema no tiene WSL 2 habilitado, ejecuta en PowerShell **como Administrador**:

```powershell
wsl --install
```

Luego reinicia tu computadora.

## Paso 3: Instalar Docker Desktop

1. Ejecuta el instalador
2. Acepta los términos y condiciones
3. Marca la opción "Use WSL 2 instead of Hyper-V" (recomendado)
4. Completa la instalación
5. Reinicia tu computadora si se solicita

## Paso 4: Verificar Instalación

Abre PowerShell y ejecuta:

```powershell
docker --version
docker compose version
```

Deberías ver las versiones instaladas.

## Paso 5: Iniciar Docker Desktop

1. Busca "Docker Desktop" en el menú de inicio
2. Inicia la aplicación
3. Espera a que se inicialice (icono de Docker en la bandeja del sistema)

## Paso 6: Ejecutar el Proyecto

Una vez Docker Desktop esté corriendo:

```powershell
docker compose up -d
```

## Nota Importante

**NO NECESITAS Docker para ejecutar los pipelines de Kedro localmente.**

Puedes ejecutar el proyecto directamente con:

```powershell
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Ejecutar pipelines
kedro run --pipeline=classification
kedro run --pipeline=regression
```

Docker solo es necesario si quieres usar Airflow para orquestación.

