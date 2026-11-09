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

## Estadística descriptiva

Cada dataset conserva una revisión y un hash de contenido. Los análisis quedan
vinculados a ambos valores para advertir cuando un resultado corresponde a una
versión anterior de la hoja.

```http
POST /api/v1/datasets/{dataset_id}/analyses/descriptive
GET  /api/v1/datasets/{dataset_id}/analyses
GET  /api/v1/analyses/{analysis_id}
```

El análisis admite columnas numéricas, de texto y fecha, separación por una
variable, percentiles, nivel de confianza y número de intervalos del histograma.
La respuesta incluye medidas clásicas y robustas, intervalos para media, mediana
y desviación, faltantes, atípicos con su fila de origen, frecuencias, histograma,
resumen de caja, Q-Q, ECDF y una prueba de normalidad Anderson–Darling. La varianza
y desviación estándar usan el estimador muestral.

## Probabilidad

El módulo de probabilidad calcula y conserva resultados para 24 distribuciones
discretas y continuas, incluyendo el catálogo observado en Minitab 15: Binomial,
Poisson, Normal, inferenciales, confiabilidad, valores extremos y una distribución
discreta personalizada. Admite masa o densidad (PDF/PMF), probabilidad acumulada (CDF),
probabilidad acumulada inversa, valores exactos, colas e intervalos. Los modos
PDF/CDF/inversa aceptan una constante o hasta 10,000 valores de una columna y
conservan errores por fila sin descartar los resultados válidos. Cada respuesta
incluye momentos teóricos, una serie para graficar, interpretación y supuestos.

```http
POST /api/v1/datasets/{dataset_id}/analyses/probability
```

Los cálculos se vinculan al espacio activo para conservar historial y
propiedad, pero no se vuelven obsoletos cuando cambia el contenido de la hoja.

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
