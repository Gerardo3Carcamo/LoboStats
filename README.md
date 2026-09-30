# LoboStats API

Backend de LoboStats, la plataforma abierta de analisis estadistico para la comunidad UAdeC.

## Desarrollo local

Requisitos: Python 3.11+ y una base MySQL 8 existente.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

La documentacion OpenAPI queda disponible en `http://localhost:8000/docs`.

En `development`, `X-User-Email` permite simular una cuenta institucional y solo acepta el dominio configurado. En produccion el API permanece cerrado hasta conectar el verificador OIDC de Google Workspace; no se debe habilitar el modo de desarrollo publicamente.

## Pruebas

```bash
pytest
ruff check .
```

## Base de datos

Configura `DATABASE_URL` con el usuario y host MySQL existentes. El contenedor no crea otro servidor MySQL:

```text
mysql+asyncmy://usuario:contrasena@host:3306/lobostats
```

Para desplegar ambos repositorios, crea una vez la red compartida y levanta primero el API:

```bash
docker network create lobostats
docker compose up -d --build
```
