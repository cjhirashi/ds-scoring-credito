import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

pio.templates.default = "plotly_dark"
BG = "#1e1e1e"
AZUL, ROJO, VERDE = "#4C9AFF", "#FF6B6B", "#5FD3A3"

RANDOM_STATE = 42
RUTA_RES = "../resultados"
RUTA_DOCS = "../docs"

df = pd.read_csv(f"{RUTA_RES}/dataset_limpio.csv")
y = df["es_malo"]
X = df.drop(columns=["ID", "es_malo"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

def layout_base(fig, titulo):
    fig.update_layout(
        title=titulo, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(size=14), margin=dict(l=60, r=30, t=60, b=50),
        height=500,
    )
    return fig

# --- 1. Distribución de la etiqueta ---
conteo = y.value_counts().rename({0: "Buen pagador", 1: "Mal pagador"})
fig1 = go.Figure(go.Bar(
    x=conteo.index, y=conteo.values, marker_color=[AZUL, ROJO],
    text=[f"{v:,} ({v/len(y)*100:.1f}%)" for v in conteo.values], textposition="outside",
))
fig1 = layout_base(fig1, f"Distribución de la etiqueta (n={len(y):,})")
fig1.update_layout(yaxis_title="Clientes")

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
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={auc:.3f})",
                           line=dict(color=AZUL, width=3), fill="tozeroy", fillcolor="rgba(76,154,255,0.15)"))
fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC=0.5)",
                           line=dict(color="gray", dash="dash")))
fig2 = layout_base(fig2, "Curva ROC — modelo final en test")
fig2.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

# --- 4. Matriz de confusión ---
cm = confusion_matrix(y_test, pred_test)
etiquetas = ["Buen pagador", "Mal pagador"]
fig3 = go.Figure(go.Heatmap(
    z=cm, x=etiquetas, y=etiquetas, colorscale="Teal",
    text=cm, texttemplate="%{text}", textfont=dict(size=20), showscale=False,
))
fig3 = layout_base(fig3, "Matriz de confusión — test")
fig3.update_layout(xaxis_title="Predicción", yaxis_title="Real", yaxis=dict(autorange="reversed"))

# --- 5. Importancia de variables ---
nombres_cat = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
nombres = list(num_cols) + list(nombres_cat)
importancias = pipe.named_steps["clf"].feature_importances_
top = pd.Series(importancias, index=nombres).sort_values(ascending=True).tail(15)
fig4 = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                         marker=dict(color=top.values, colorscale="Teal")))
fig4 = layout_base(fig4, "Top 15 variables más importantes")
fig4.update_layout(xaxis_title="Importancia", height=600)

# --- Página consolidada ---
html_partes = []
for fig, descripcion in [
    (fig1, "Solo 1.7% de los clientes cruzados son mal pagador: fuerte desbalance de clases."),
    (fig2, "El modelo discrimina mal pagador de buen pagador con ROC-AUC 0.814."),
    (fig3, "6,968 buenos pagadores bien clasificados; detecta 45 de 123 malos pagadores del test."),
    (fig4, "Edad, ingreso total y antigüedad laboral concentran la mayor señal predictiva."),
]:
    bloque = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
    html_partes.append(f'<div class="grafica"><p class="descripcion">{descripcion}</p>{bloque}</div>')

pagina = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Scoring Crediticio — Gráficas interactivas</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background:{BG}; color:#eee; font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 30px 20px; }}
  h1 {{ font-size: 1.6rem; }}
  p.intro {{ color:#aaa; }}
  .descripcion {{ color:#ccc; font-size: 0.95rem; margin-bottom: 4px; }}
  .grafica {{ margin-bottom: 36px; }}
  a {{ color:{AZUL}; }}
</style>
</head>
<body>
<h1>Scoring Crediticio — Predicción de Riesgo de Impago</h1>
<p class="intro">Gráficas interactivas del análisis. Repo completo: <a href="https://github.com/cjhirashi/ds-scoring-credito">github.com/cjhirashi/ds-scoring-credito</a></p>
{''.join(html_partes)}
</body>
</html>"""

with open(f"{RUTA_DOCS}/index.html", "w", encoding="utf-8") as f:
    f.write(pagina)

print("página interactiva generada en docs/index.html")
