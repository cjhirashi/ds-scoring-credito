# ds-scoring-credito

## Pregunta

¿Puede predecirse, a partir de los datos de solicitud de una tarjeta de crédito (ingreso, empleo, vivienda, composición familiar), si un cliente se convertirá en mal pagador (atraso ≥60 días en algún momento de su historial)?

## Datos

Kaggle, [`rikdifos/credit-card-approval-prediction`](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction):

- `application_record.csv` (438,557 filas): datos de la solicitud — demografía, ingreso, ocupación, vivienda.
- `credit_record.csv` (1,048,575 filas): historial mensual de pago por cliente (`STATUS`), de -60 a 0 meses.

Solo 36,457 de los 438,510 clientes de `application_record` tienen historial en `credit_record` y pueden etiquetarse; el resto se descarta.

**Etiqueta:** mal pagador = `STATUS` en {2,3,4,5} (atraso ≥60 días) en algún mes de su historial. Con este criterio: 667 de 45,985 clientes con historial (1.45%) son mal pagador — fuerte desbalance de clases.

## Cómo se reproduce

`datos/` no está versionado (son datos crudos de Kaggle, redistribuibles, regenerables). Antes de correr los scripts:

```
kaggle datasets download rikdifos/credit-card-approval-prediction -p datos --unzip
```

```
cd trabajo
python 01-eda.py        # EDA: nulos, duplicados, desbalance, genera resultados/etiqueta_riesgo.csv
python 02-limpieza.py   # limpieza y feature engineering, genera resultados/dataset_limpio.csv
python 03-modelo.py     # baseline + comparación de algoritmos, genera resultados/comparacion_modelos.csv
python 04-seleccion.py  # GridSearchCV + evaluación en test, genera resultados/modelo_final.csv
```

Entorno: `conda activate py310` (Miniforge en `/mnt/disco2/cjhirashi-data/data-science/_tools/miniforge3`).

## Resultado

Random Forest (`n_estimators=400`, `max_depth=None`, `min_samples_leaf=1`, `class_weight="balanced"`), seleccionado por validación cruzada 5-fold sobre ROC-AUC:

| Métrica | CV (train) | Test (nunca visto) |
|---|---|---|
| ROC-AUC | 0.7993 | 0.8144 |
| F1 | — | 0.2439 |
| Precision | — | 0.1829 |
| Recall | — | 0.3659 |

El modelo discrimina mal pagador de buen pagador con ROC-AUC ~0.81, pero con precisión baja (18%): de cada 5 clientes marcados como riesgo, 1 realmente lo es. Útil como filtro de revisión manual, no para una decisión automática de rechazo.

Random Forest superó claramente a los modelos lineales probados (regresión logística ROC-AUC 0.57, árbol de decisión simple 0.55), ambos apenas por encima del baseline dummy (0.50).

## Limitaciones

- El test tiene solo 123 positivos: las métricas de precision/recall/F1 tienen varianza alta con ese tamaño de muestra.
- La etiqueta asume que "sin atraso ≥60 días en los últimos 60 meses" equivale a "buen pagador" — ignora que un cliente con poco historial (pocos meses observados) puede simplemente no haber tenido oportunidad de atrasarse.
- No se probaron técnicas de remuestreo (SMOTE/undersampling) ni modelos de gradient boosting; `class_weight="balanced"` fue la única estrategia de desbalance usada.
