# CI/CD del backend en Raspberry Pi 5

El workflow `.github/workflows/backend-ci-cd.yml` contiene dos trabajos:

1. **Lint, test and build** usa Python 3.11, valida Ruff, aplica todas las migraciones sobre una
   base SQLite vacía, ejecuta las pruebas y construye el wheel.
2. **Deploy to Raspberry Pi 5** se ejecuta después del CI para `develop-tesis` y llama a
   `/opt/devops-scripts/Python/deploy-lobostats.sh`.

El job entrega al script un repositorio Git local con exactamente `${GITHUB_SHA}`. No requiere
una deploy key permanente ni expone credenciales del repositorio.

## 1. Runner

Se puede reutilizar el runner del frontend si tiene estas etiquetas:

```text
self-hosted, Linux, ARM64, lobostats-rpi
```

El servicio debe ejecutarse como el usuario que despliega actualmente, no como `root`. La RPi
necesita Python 3.11 o posterior con el módulo `venv`, además de Git, rsync, curl, jq y systemd.

## 2. Configuración privada del backend

El archivo real permanece en:

```text
/etc/lobostats/lobostats-api.env
```

Debe existir antes del despliegue, tener permisos restrictivos y contener como mínimo una
`DATABASE_URL` válida. No copies este archivo, la contraseña MySQL ni otros secretos a GitHub.

Comprueba la configuración:

```bash
sudo test -f /etc/lobostats/lobostats-api.env
sudo grep -q CAMBIA_ESTA_PASSWORD /etc/lobostats/lobostats-api.env && echo "Falta configurar"
```

Mientras no se implemente el verificador OIDC de Google Workspace, `APP_ENV=development` solo
debe exponerse detrás de Cloudflare Access o una VPN. No lo presentes como autenticación de
producción.

## 3. Ejecución no interactiva

Prueba el script con el mismo usuario del servicio del runner:

```bash
bash /opt/devops-scripts/Python/deploy-lobostats.sh \
  lobostats-api \
  /ruta/a/un/clone-local-de-LoboStats \
  6100 \
  app.main:app \
  develop-tesis \
  https://lobostats.arechsolutions.com
```

GitHub Actions no puede responder una contraseña de `sudo`. Concede solamente los permisos
administrativos necesarios para Nginx, systemd y los directorios de despliegue. No ejecutes el
runner como `root` ni configures `NOPASSWD: ALL`.

El script debe ser legible por el usuario del runner:

```bash
chmod 750 /opt/devops-scripts/Python/deploy-lobostats.sh
```

## 4. GitHub

Crea o reutiliza el environment **production**. Si deseas aprobación manual, configura un
revisor requerido. Protege `develop-tesis` exigiendo el check **Lint, test and build**.

## Activación

- Pull request hacia `develop-tesis`: solo CI.
- Push a `develop-tesis`: CI y despliegue.
- Ejecución manual: permite decidir si se despliega.

El job exige que el servicio quede activo y que `/api/v1/health` responda localmente y a través
de `https://lobostats.arechsolutions.com`. Los despliegues se serializan para evitar dos
reinicios o migraciones simultáneos.
