# LoboStats Backend

Backend para carga temporal de datos, estadística descriptiva, inferencia, potencia, ANOVA, distribuciones y datos listos para gráficos. Los datasets se guardan **solo en memoria**: no hay base de datos y todo se pierde al reiniciar el proceso.

## Requisitos e instalación

- Python 3.11

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

La API queda disponible en `http://localhost:5000`. La documentación interactiva está en `/api/docs` y el contrato en `/api/openapi.yaml`.

## Docker

```bash
docker build -t lobostats-api .
docker run --rm -p 5000:5000 lobostats-api
```

## API

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/health` | Estado del servicio |
| POST | `/api/data/upload` | Carga CSV, TSV, XLS, XLSX o JSON mediante multipart `file` |
| POST | `/api/data/manual` | Crea dataset con `{name, columns, rows}` |
| GET | `/api/data/{dataset_id}` | Obtiene filas paginadas con `page` y `page_size` |
| DELETE | `/api/data/{dataset_id}` | Elimina un dataset |
| POST | `/api/descriptive/summary` | Resumen por columnas y Shapiro-Wilk |
| POST | `/api/descriptive/frequencies` | Tabla de frecuencias |
| POST | `/api/charts/histogram` | Intervalos y cuentas de histograma |
| POST | `/api/charts/boxplot` | Cuartiles, bigotes y atípicos |
| POST | `/api/charts/scatter` | Pares numéricos para dispersión |
| POST | `/api/hypothesis/t-test` | t de una muestra, independiente o pareada |
| POST | `/api/hypothesis/anova` | ANOVA de una vía |
| POST | `/api/hypothesis/chi-square` | Chi-cuadrado de independencia |
| POST | `/api/hypothesis/correlation` | Correlación de Pearson |
| POST | `/api/basic-stats/confidence-interval`, `/z-test`, `/by-group`, `/correlation-matrix`, `/covariance-matrix` | Intervalos, prueba Z, grupos y matrices |
| POST | `/api/variance/normality`, `/levene`, `/bartlett` | Normalidad y homogeneidad de varianzas |
| POST | `/api/nonparametric/mann-whitney`, `/wilcoxon`, `/kruskal-wallis`, `/friedman`, `/sign-test` | Pruebas no paramétricas |
| POST | `/api/proportion/one-sample`, `/two-sample` | Pruebas z e IC de proporciones |
| POST | `/api/correlation/spearman`, `/kendall`, `/partial` | Correlaciones no paramétricas y parcial |
| POST | `/api/power/t-test`, `/anova`, `/sample-size` | Potencia y tamaño muestral |
| POST | `/api/anova/one-way`, `/two-way`, `/post-hoc`, `/repeated-measures`, `/main-effects`, `/interaction` | ANOVA con statsmodels |
| POST | `/api/charts/bar`, `/pie`, `/pareto`, `/qq`, `/violin`, `/dot`, `/line`, `/area`, `/interval`, `/pairplot`, `/bubble`, `/heatmap`, `/scatter-3d` | Datos para gráficos avanzados |
| POST | `/api/distributions/pdf`, `/cdf`, `/sf`, `/ppf`, `/interval`, `/random`, `/curve` | Evaluación, simulación y curvas de distribuciones |
| POST | `/api/distributions/fit`, `/goodness-of-fit`, `/identify`, `/probability-plot` | Ajuste MLE, identificación y gráficos de probabilidad |

### Esquemas de t-test

```json
{"dataset_id":"...","type":"one-sample","column":"value","popmean":10,"alpha":0.05}
{"dataset_id":"...","type":"independent","value_column":"value","group_column":"group","groups":["A","B"]}
{"dataset_id":"...","type":"paired","column1":"before","column2":"after"}
```

`alpha` es opcional y vale `0.05` por defecto. La t independiente usa la corrección de Welch. Las respuestas estadísticas incluyen `results` e `interpretation` en español. Los valores no finitos se serializan como `null` para producir JSON válido.

### Payloads estadísticos

Todos los endpoints que operan sobre datos reciben `dataset_id`. Ejemplos representativos:

```json
{"dataset_id":"...","column":"value","statistic":"median","confidence":0.95,"resamples":5000,"seed":0}
{"dataset_id":"...","column":"value","null_mean":10,"population_std":2.5,"alternative":"two-sided"}
{"dataset_id":"...","columns":["x","y"],"method":"spearman"}
{"dataset_id":"...","column":"value","method":"dagostino","alpha":0.05}
{"dataset_id":"...","value_column":"value","group_column":"group"}
{"dataset_id":"...","column1":"before","column2":"after"}
{"dataset_id":"...","columns":["before","after","value"]}
{"successes":60,"nobs":100,"null_proportion":0.5,"alternative":"two-sided"}
{"successes":[60,45],"nobs":[100,100]}
{"dataset_id":"...","x":"x","y":"y","controls":["age"]}
{"effect_size":0.5,"power":0.8,"design":"t-test-independent"}
{"test":"anova","effect_size":0.25,"power":0.8,"groups":3}
```

Normalidad acepta `shapiro`, `anderson`, `kolmogorov-smirnov` y `dagostino`. Las alternativas de proporciones son `two-sided`, `smaller` y `larger`. En potencia debe enviarse exactamente uno de `power` o `sample_size`; `/sample-size` usa potencia `0.8` por defecto.

### Payloads ANOVA

```json
{"dataset_id":"...","value_column":"value","group_column":"group"}
{"dataset_id":"...","value_column":"value","factor_a":"treatment","factor_b":"sex","interaction":true,"typ":2}
{"dataset_id":"...","value_column":"value","group_column":"group","method":"scheffe"}
{"dataset_id":"...","subject":"subject","within":"time","value":"score"}
```

`typ` acepta `1`, `2` o `3`. Post-hoc acepta `tukey`, `bonferroni` o `scheffe`. Medidas repetidas exige una observación por combinación sujeto/nivel y un diseño completo balanceado. `/main-effects` e `/interaction` usan el mismo payload de dos vías.

### Payloads de gráficos y límites

`bar`, `pie` y `pareto` reciben `category`, `value` opcional y `aggregation` (`sum`, `mean`, `median`, `count`). `qq`, `dot` y `violin` reciben `column`; violin admite `group_by`. `line`/`area` reciben `x,y`; interval recibe `value,group`; bubble recibe `x,y,size` y `color` opcional; scatter-3d recibe `x,y,z`. Pairplot y heatmap reciben `columns` (heatmap admite `method`). Histograma admite `density` y `kde` booleanos.

Las series se limitan a 5.000 puntos mediante selección determinista; pairplot a 1.000 filas y 10 columnas; matrices a 20 columnas; violines a 1.000 valores por grupo y categorías a 100. La respuesta informa muestreo y límites cuando aplica.

### Distribuciones e identificación

Las operaciones directas reciben `distribution` y `params`. `pdf` usa PMF para distribuciones discretas. `x` y `q` admiten un escalar o una lista de hasta 10.000 valores; `random` genera como máximo 10.000 valores y `curve` como máximo 2.000 puntos.

```json
{"distribution":"normal","params":{"mean":0,"std":1},"x":[-1,0,1]}
{"distribution":"binomial","params":{"n":10,"p":0.3},"x":3}
{"distribution":"t","params":{"df":8,"loc":0,"scale":1},"q":0.975}
{"distribution":"gamma","params":{"shape":2,"loc":0,"scale":3},"confidence":0.95}
{"distribution":"poisson","params":{"mu":4},"size":100,"seed":7}
{"distribution":"normal","params":{"mean":0,"std":1},"x_min":-4,"x_max":4,"points":300}
```

Se soportan `normal(mean,std)`, `t(df,loc,scale)`, `chi2(df,loc,scale)`, `f(dfn,dfd,loc,scale)`, `binomial(n,p,loc)`, `poisson(mu,loc)`, `hypergeom(M,n,N,loc)`, `exponential(loc,scale)`, `gamma(shape,loc,scale)`, `beta(a,b,loc,scale)`, `weibull(shape,loc,scale)`, `lognormal(shape,loc,scale)`, `uniform(loc,scale)`, `logistic(loc,scale)`, `gumbel_l(loc,scale)` y `gumbel_r(loc,scale)`. `loc` vale 0 y `scale` vale 1 cuando se omiten; los parámetros de forma obligatorios no tienen valor predeterminado.

Distribution ID solo admite candidatas continuas. Por defecto compara `normal`, `lognormal`, `weibull`, `exponential`, `gamma`, `logistic`, `gumbel_l` y `gumbel_r`:

```json
{"dataset_id":"...","column":"value","distribution":"normal","bootstrap_reps":200}
{"dataset_id":"...","column":"value","candidates":["normal","lognormal","weibull"],"bootstrap_reps":200}
{"dataset_id":"...","column":"value","distribution":"normal","confidence":0.95}
```

El ajuste usa MLE. Cuando todos los datos son positivos, `lognormal`, `weibull`, `exponential` y `gamma` fijan `loc=0` para respetar el soporte y usar la parametrización reducida; `weibull-3p` y `exponential-2p` dejan `loc` libre. Se procesan como máximo 5.000 datos mediante selección determinista. El p-value de Anderson-Darling usa bootstrap paramétrico reproducible (200 réplicas por defecto, de 20 a 1.000); `identify` aísla fallos por candidata y ordena por AD, con AIC como desempate. Los probability plots usan hasta 1.000 datos y bandas puntuales Beta para estadísticos de orden, no bandas simultáneas.

### Consideraciones estadísticas

- El KS normaliza con parámetros estimados, por lo que su p-valor es aproximado y no sustituye Lilliefors.
- El KS de Distribution ID también es aproximado por estimar parámetros en la misma muestra. Chi-cuadrado usa bins por cuantiles y ajusta los grados de libertad; no se informa cuando las frecuencias esperadas o los grados de libertad son insuficientes.
- Bartlett presupone normalidad; Levene con mediana es más robusto.
- Los IC de proporción individual son Wilson; el IC de diferencia usa aproximación normal no agrupada.
- ANOVA OLS presupone independencia, residuos aproximadamente normales y homocedasticidad. El diagnóstico incluido no demuestra esos supuestos.
- AnovaRM de statsmodels no aplica correcciones de esfericidad ni admite diseños incompletos.
- Potencia requiere tamaños de efecto estandarizados definidos antes del análisis.

## Pruebas

```powershell
python -m pytest -q
```

El archivo `sample_data/iris.csv` permite probar rápidamente la carga multipart.
