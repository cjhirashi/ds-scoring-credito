# 💳 Scoring Crediticio — Predicción de Riesgo de Impago

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-orange?logo=scikitlearn)](https://scikit-learn.org/stable/)
[![Conda](https://img.shields.io/badge/Conda-Env%20Management-44A833?logo=anaconda)](https://docs.conda.io/)

---

## 🚀 Descripción

Proyecto propio de portafolio. Desarrollo de un modelo de **Machine Learning** para predecir si un solicitante de tarjeta de crédito se convertirá en **mal pagador** (atraso ≥60 días en su historial de pagos), usando el dataset de Kaggle [`rikdifos/credit-card-approval-prediction`](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction).

---

## ✨ Objetivos principales

* Cruzar los datos de solicitud (`application_record`) con el historial de pago (`credit_record`) y construir la etiqueta de riesgo.
* Explorar el desbalance de clases de la variable objetivo (mal pagador vs. buen pagador).
* Limpiar nulos y transformar variables de edad/antigüedad laboral codificadas como días negativos.
* Entrenar y comparar modelos de clasificación con corrección de desbalance (`class_weight="balanced"`).
* Seleccionar el mejor modelo por validación cruzada y afinar hiperparámetros.
* Evaluar el modelo final en un conjunto de prueba nunca visto.

---

## 🧰 Tecnologías utilizadas

* [Python 3.10](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Conda](https://docs.conda.io/) (Miniforge) — gestión de entornos

---

## ✅ Pasos del proyecto

El análisis se llevó a cabo en scripts numerados dentro de `trabajo/`, cada uno ejecutable de forma independiente:

1. **`01-eda.py` — Carga y EDA**
   - Inspección de filas, columnas, duplicados y nulos en ambos CSV.
   - Construcción de la etiqueta de riesgo cruzando `application_record` con `credit_record`.
   - Definición de "mal pagador": `STATUS` en {2,3,4,5} (atraso ≥60 días) en algún mes del historial.
   - Medición del desbalance de clases.

2. **`02-limpieza.py` — Limpieza y feature engineering**
   - Eliminación de IDs duplicados.
   - Nulos de `OCCUPATION_TYPE` → categoría `Sin_declarar`.
   - Transformación de `DAYS_BIRTH`/`DAYS_EMPLOYED` (negativos, con centinela para desempleados) en `EDAD_ANIOS`, `ANIOS_EMPLEO` y bandera `DESEMPLEADO`.

3. **`03-modelo.py` — Baseline y comparación de algoritmos**
   - Partición estratificada train/test (80/20).
   - Entrenamiento de baseline *dummy*, regresión logística, árbol de decisión y Random Forest, todos con `class_weight="balanced"`.
   - Comparación por ROC-AUC, F1, precision y recall.

4. **`04-seleccion.py` — Selección y evaluación final**
   - `GridSearchCV` con validación cruzada estratificada de 5 folds sobre el modelo ganador (Random Forest).
   - Evaluación del mejor modelo en el conjunto de prueba, nunca visto durante el ajuste.

---

## ⚙️ Estructura del proyecto

```bash
.
├── datos/                          # Dataset crudo de Kaggle (no versionado, ver "Instalación y uso")
│   ├── application_record.csv
│   └── credit_record.csv
├── trabajo/
│   ├── 01-eda.py
│   ├── 02-limpieza.py
│   ├── 03-modelo.py
│   └── 04-seleccion.py
├── resultados/
│   ├── etiqueta_riesgo.csv
│   ├── dataset_limpio.csv
│   ├── comparacion_modelos.csv
│   └── modelo_final.csv
├── .gitignore
└── README.md
```

---

## 📑 Dataset

**Fuente:** Kaggle, [`rikdifos/credit-card-approval-prediction`](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (licencia CC0).

**`application_record.csv`** (438,557 filas) — datos de la solicitud:

* `ID` → identificador de cliente
* `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY` → demografía y posesiones
* `CNT_CHILDREN`, `CNT_FAM_MEMBERS` → composición familiar
* `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE` → ingreso
* `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE` → perfil socioeconómico
* `DAYS_BIRTH`, `DAYS_EMPLOYED` → edad y antigüedad laboral (en días negativos)
* `OCCUPATION_TYPE` → ocupación (30% nulo)

**`credit_record.csv`** (1,048,575 filas) — historial mensual de pago:

* `ID` → identificador de cliente
* `MONTHS_BALANCE` → mes relativo al presente (0 a -60)
* `STATUS` → estado de pago ese mes (`C`=pagado, `X`=sin préstamo ese mes, `0`-`5`=días de atraso crecientes)

Solo **36,457 de 438,510** clientes de `application_record` tienen historial en `credit_record` y pueden etiquetarse; el resto se descarta por falta de información de pago.

---

## 📊 Resultados

### 🔹 Comparación de modelos base (test, split simple)

Con la etiqueta de riesgo (1.45% mal pagador, fuerte desbalance) y `class_weight="balanced"`:

| Modelo | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Random Forest** | **0.7946** | 0.2439 | 0.1829 | 0.3659 |
| Regresión logística | 0.5659 | 0.0414 | 0.0216 | 0.4878 |
| Árbol de decisión | 0.5521 | 0.0401 | 0.0209 | 0.5041 |
| Baseline dummy | 0.5000 | 0.0 | 0.0 | 0.0 |

Random Forest superó claramente a los modelos lineales, que apenas se despegaron del baseline.

### 🚀 Modelo final en conjunto de prueba

Seleccionado por `GridSearchCV` con validación cruzada 5-fold sobre ROC-AUC: **Random Forest** (`n_estimators=400`, `max_depth=None`, `min_samples_leaf=1`, `class_weight="balanced"`).

| Métrica | CV (train) | Test (nunca visto) |
|---|---|---|
| ROC-AUC | 0.7993 | **0.8144** |
| F1 | — | 0.2439 |
| Precision | — | 0.1829 |
| Recall | — | 0.3659 |

El resultado en CV y test es consistente, sin señal fuerte de sobreajuste.

### ✅ Conclusión general

El modelo discrimina mal pagador de buen pagador con ROC-AUC ~0.81, pero con precisión baja (18%): de cada 5 clientes marcados como riesgo, solo 1 realmente lo es. Es útil como **filtro de revisión manual** para priorizar casos sospechosos, no para una decisión automática de rechazo de crédito.

---

## ⚠️ Limitaciones

* El conjunto de prueba tiene solo 123 positivos: las métricas de precision/recall/F1 tienen varianza alta con ese tamaño de muestra.
* La etiqueta asume que "sin atraso ≥60 días en los últimos 60 meses" equivale a "buen pagador" — ignora que un cliente con poco historial observado puede simplemente no haber tenido oportunidad de atrasarse.
* No se probaron técnicas de remuestreo (SMOTE/undersampling) ni modelos de gradient boosting; `class_weight="balanced"` fue la única estrategia de desbalance usada.

---

## ▶️ Instalación y uso

1. **Clonar repositorio**

   ```bash
   git clone git@github.com:cjhirashi/ds-scoring-credito.git
   cd ds-scoring-credito
   ```

2. **Descargar el dataset** (no está versionado en el repo)

   ```bash
   kaggle datasets download rikdifos/credit-card-approval-prediction -p datos --unzip
   ```

3. **Activar el entorno con Conda**

   ```bash
   conda activate py310
   ```

4. **Ejecutar los scripts en orden**

   ```bash
   cd trabajo
   python 01-eda.py
   python 02-limpieza.py
   python 03-modelo.py
   python 04-seleccion.py
   ```

---

## 👨‍💻 Autor

**Carlos Jiménez Hirashi**
💼 Data Scientist Jr. | Python & Machine Learning
📧 [cjhirashi@gmail.com](mailto:cjhirashi@gmail.com) · 🌐 [LinkedIn](https://www.linkedin.com/in/cjhirashi)
