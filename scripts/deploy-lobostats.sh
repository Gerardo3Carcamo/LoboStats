#!/usr/bin/env bash
set -Eeuo pipefail

if ((EUID == 0)); then
  echo "❌ Ejecuta este script como usuario normal, no con sudo"
  exit 1
fi

APP_NAME="${1:-lobostats-api}"
REPO="${2:-}"
PORT="${3:-6100}"
UVICORN_MODULE="${4:-app.main:app}"
BRANCH="${5:-develop-tesis}"
PUBLIC_ORIGIN="${6:-https://lobostats.arechsolutions.com}"

if [[ -z "$REPO" ]]; then
  read -rp "Repositorio GitHub del backend: " REPO
fi

if [[ ! "$APP_NAME" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]]; then
  echo "❌ Nombre de aplicación inválido"
  exit 1
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
  echo "❌ Puerto inválido"
  exit 1
fi

if [[ ! "$UVICORN_MODULE" =~ ^[A-Za-z0-9_.]+:[A-Za-z0-9_]+$ ]]; then
  echo "❌ Módulo Uvicorn inválido"
  exit 1
fi

for command in git rsync python3 curl systemctl ss; do
  command -v "$command" >/dev/null || {
    echo "❌ Falta instalar: $command"
    exit 1
  }
done

DEPLOY_USER="$USER"
REPO_ROOT="/var/repos/python-apis"
APP_ROOT="/var/www/python-apis"
REPO_DIR="$REPO_ROOT/$APP_NAME"
APP_DIR="$APP_ROOT/$APP_NAME"
VENV_DIR="$APP_DIR/venv"
ENV_DIR="/etc/lobostats"
ENV_FILE="$ENV_DIR/$APP_NAME.env"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

echo "🚀 Desplegando $APP_NAME"

sudo install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$REPO_ROOT" "$APP_ROOT"
sudo install -d -m 700 "$ENV_DIR"

if ! sudo test -f "$ENV_FILE"; then
  sudo tee "$ENV_FILE" >/dev/null <<EOF
APP_NAME="LoboStats API"
APP_ENV=development
API_V1_PREFIX=/api/v1
DATABASE_URL=mysql+asyncmy://lobostats:CAMBIA_ESTA_PASSWORD@127.0.0.1:3306/lobostats
ALLOWED_EMAIL_DOMAIN=uadec.edu.mx
DEV_USER_EMAIL=alumno@uadec.edu.mx
CORS_ORIGINS=$PUBLIC_ORIGIN
EOF

  sudo chmod 600 "$ENV_FILE"

  echo "⚠️ Se creó $ENV_FILE"
  echo "Edita DATABASE_URL y vuelve a ejecutar el script:"
  echo "sudo nano $ENV_FILE"
  exit 1
fi

if sudo grep -q "CAMBIA_ESTA_PASSWORD" "$ENV_FILE"; then
  echo "❌ Debes configurar DATABASE_URL en $ENV_FILE"
  exit 1
fi

echo "📦 Clonando rama $BRANCH..."

git clone \
  --depth 1 \
  --branch "$BRANCH" \
  -- "$REPO" "$TMP_DIR/source"

sudo install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$REPO_DIR" "$APP_DIR"

sudo rsync -a --delete \
  --exclude=".git" \
  -- "$TMP_DIR/source/" "$REPO_DIR/"

sudo rsync -a --delete \
  --exclude=".git" \
  --exclude=".env" \
  --exclude="venv" \
  --chown="$DEPLOY_USER:$DEPLOY_USER" \
  -- "$TMP_DIR/source/" "$APP_DIR/"

echo "🐍 Creando entorno virtual..."

# Stop the managed unit even when it is activating or already in a restart loop.
if sudo systemctl cat "$APP_NAME.service" >/dev/null 2>&1; then
  echo "⏹️ Deteniendo servicio anterior..."
  sudo systemctl stop "$APP_NAME"
fi

# A listener at this point is not managed by this deployment unit. Refuse to
# hide the collision behind a health check from the wrong application.
PORT_LISTENER="$(sudo ss -H -ltnp "sport = :$PORT" || true)"
if [[ -n "$PORT_LISTENER" ]]; then
  echo "❌ El puerto $PORT sigue ocupado por un proceso ajeno a $APP_NAME:"
  echo "$PORT_LISTENER"
  echo "Elimina su mecanismo de arranque antes de volver a desplegar."
  exit 1
fi

# A previous deployment may have created the environment as root. Removing it
# through sudo is safe here because VENV_DIR is derived from the validated app name.
sudo rm -rf -- "$VENV_DIR"
python3 -m venv "$VENV_DIR"

"$VENV_DIR/bin/pip" install --upgrade pip wheel

if [[ -f "$APP_DIR/pyproject.toml" ]]; then
  "$VENV_DIR/bin/pip" install "$APP_DIR"
elif [[ -f "$APP_DIR/requirements.txt" ]]; then
  "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
else
  echo "❌ No se encontró pyproject.toml ni requirements.txt"
  exit 1
fi

echo "⚙️ Configurando systemd..."

sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=LoboStats FastAPI Backend
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStartPre=$VENV_DIR/bin/alembic upgrade head
ExecStart=$VENV_DIR/bin/uvicorn $UVICORN_MODULE --host 127.0.0.1 --port $PORT
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=30
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl reset-failed "$APP_NAME" || true
sudo systemctl enable "$APP_NAME"
sudo systemctl restart "$APP_NAME"

echo "🔍 Validando API..."

API_READY=false

for _ in {1..30}; do
  if curl --fail --silent \
    "http://127.0.0.1:$PORT/api/v1/health" >/dev/null; then
    API_READY=true
    break
  fi

  sleep 1
done

if [[ "$API_READY" != true ]]; then
  echo "❌ La API no respondió"
  sudo systemctl status "$APP_NAME" --no-pager || true
  sudo journalctl -u "$APP_NAME" -n 100 --no-pager || true
  exit 1
fi

echo
echo "======================================="
echo "✅ BACKEND DESPLEGADO"
echo "🔒 http://127.0.0.1:$PORT"
echo "📋 Logs: sudo journalctl -u $APP_NAME -f"
echo "======================================="
echo
echo "⚠️ APP_ENV=development solo debe usarse detrás de Cloudflare Access o VPN."
