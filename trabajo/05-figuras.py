import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix

RANDOM_STATE = 42
RUTA_RES = "../resultados"

df = pd.read_csv(f"{RUTA_RES}/dataset_limpio.csv")
y = df["es_malo"]
X = df.drop(columns=["ID", "es_malo"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# --- 1. Distribución de la etiqueta (desbalance) ---
fig, ax = plt.subplots(figsize=(5, 4))
y.value_counts().rename({0: "Buen pagador", 1: "Mal pagador"}).plot(
    kind="bar", color=["#4C72B0", "#C44E52"], ax=ax
)
ax.set_title(f"Distribución de la etiqueta (n={len(y)})")
ax.set_ylabel("Clientes")
for i, v in enumerate(y.value_counts()):
    ax.text(i, v, f"{v}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_01_distribucion_etiqueta.png", dpi=120)
plt.close()

# --- 2. Entrenar modelo final (hiperparámetros ya seleccionados en 04-seleccion.py) ---
preprocesador = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
modelo = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)
pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
pipe.fit(X_train, y_train)

proba_test = pipe.predict_proba(X_test)[:, 1]
pred_test = pipe.predict(X_test)

# --- 3. Curva ROC ---
fpr, tpr, _ = roc_curve(y_test, proba_test)
auc = roc_auc_score(y_test, proba_test)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f"Random Forest (AUC={auc:.3f})", color="#4C72B0")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC=0.5)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curva ROC — modelo final en test")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_02_curva_roc.png", dpi=120)
plt.close()

# --- 4. Matriz de confusión ---
fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_test, pred_test)
ConfusionMatrixDisplay(cm, display_labels=["Buen pagador", "Mal pagador"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Matriz de confusión — test")
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_03_matriz_confusion.png", dpi=120)
plt.close()

# --- 5. Importancia de variables ---
nombres_cat = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
nombres = list(num_cols) + list(nombres_cat)
importancias = pipe.named_steps["clf"].feature_importances_
top = pd.Series(importancias, index=nombres).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 6))
top.sort_values().plot(kind="barh", color="#55A868", ax=ax)
ax.set_title("Top 15 variables más importantes — Random Forest")
ax.set_xlabel("Importancia")
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_04_importancia_variables.png", dpi=120)
plt.close()

print("figuras generadas en resultados/: fig_01 a fig_04")
