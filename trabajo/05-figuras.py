import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

BG = "#1e1e1e"
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#444", "grid.color": "#333",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "#ccc", "ytick.color": "#ccc",
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
})
PALETA = sns.color_palette("mako", as_cmap=False)
AZUL, ROJO, VERDE = "#4C9AFF", "#FF6B6B", "#5FD3A3"

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

# --- 1. Distribución de la etiqueta ---
conteo = y.value_counts().rename({0: "Buen pagador", 1: "Mal pagador"})
fig, ax = plt.subplots(figsize=(6, 5))
barras = ax.bar(conteo.index, conteo.values, color=[AZUL, ROJO], width=0.55, edgecolor="none")
for b, v in zip(barras, conteo.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(y)*100:.1f}%)",
            ha="center", va="bottom", fontsize=12, color="white")
ax.set_title(f"Distribución de la etiqueta (n={len(y):,})")
ax.set_ylabel("Clientes")
ax.set_ylim(0, conteo.max() * 1.2)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_01_distribucion_etiqueta.png", dpi=150)
plt.close()

# --- 2. Entrenar modelo final ---
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
fig, ax = plt.subplots(figsize=(6, 6))
ax.fill_between(fpr, tpr, color=AZUL, alpha=0.15)
ax.plot(fpr, tpr, color=AZUL, linewidth=3, label=f"Random Forest (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1.5, label="Azar (AUC=0.5)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curva ROC — modelo final en test")
ax.legend(loc="lower right", frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_02_curva_roc.png", dpi=150)
plt.close()

# --- 4. Matriz de confusión ---
cm = confusion_matrix(y_test, pred_test)
etiquetas = ["Buen pagador", "Mal pagador"]
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
            xticklabels=etiquetas, yticklabels=etiquetas,
            annot_kws={"size": 18, "weight": "bold"}, linewidths=1, linecolor=BG, ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusión — test")
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_03_matriz_confusion.png", dpi=150)
plt.close()

# --- 5. Importancia de variables ---
nombres_cat = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
nombres = list(num_cols) + list(nombres_cat)
importancias = pipe.named_steps["clf"].feature_importances_
top = pd.Series(importancias, index=nombres).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
colores = sns.color_palette("mako", n_colors=len(top))[::-1]
ax.barh(top.index[::-1], top.values[::-1], color=colores)
ax.set_title("Top 15 variables más importantes", fontsize=16)
ax.set_xlabel("Importancia")
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(f"{RUTA_RES}/fig_04_importancia_variables.png", dpi=150)
plt.close()

print("figuras generadas en resultados/ con seaborn: fig_01 a fig_04")
