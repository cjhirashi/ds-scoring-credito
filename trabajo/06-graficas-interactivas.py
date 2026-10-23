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
        autosize=True,
        xaxis=dict(automargin=True), yaxis=dict(automargin=True),
    )
    fig._alto_ref = 500
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
fig4.update_layout(xaxis_title="Importancia")
fig4._alto_ref = 600

# --- KPIs resumen ---
kpis = [
    ("ROC-AUC (test)", f"{auc:.3f}", "Capacidad del modelo para separar mal pagador de buen pagador. 0.5 = azar, 1.0 = perfecto.", AZUL),
    ("Desbalance", f"{(y.mean()*100):.1f}%", "Proporción de clientes mal pagador sobre el total con historial — clase minoritaria rara.", ROJO),
    ("Recall", "36.6%", "De cada 10 malos pagadores reales, el modelo detecta ~4 antes de que incumplan.", VERDE),
    ("Precision", "18.3%", "De cada 10 alertas del modelo, ~2 son realmente mal pagador — útil como filtro, no como veredicto.", "#F5A623"),
]
kpi_html = "".join(f'''
  <div class="kpi" style="border-top: 3px solid {color}">
    <div class="kpi-valor" style="color:{color}">{valor}</div>
    <div class="kpi-label">{label}</div>
    <div class="kpi-nota">{nota}</div>
  </div>''' for label, valor, nota, color in kpis)

# --- Página consolidada ---
graficas = [
    (fig1, "1 · El problema: clases muy desbalanceadas",
     "Solo 1.7% de los 36,457 clientes con historial cruzado son mal pagador (atraso ≥60 días). "
     "Esto obliga a usar <code>class_weight=\"balanced\"</code> y a medir con ROC-AUC/recall en vez de accuracy, "
     "que con este desbalance sería engañosamente alta incluso prediciendo siempre \"buen pagador\". "
     "<strong>Pasa el mouse sobre cada barra</strong> para ver el conteo exacto."),
    (fig2, "2 · Qué tan bien discrimina el modelo",
     "La curva ROC compara la tasa de verdaderos positivos contra falsos positivos en todos los umbrales de decisión posibles. "
     "Entre más se aleje la curva azul de la diagonal gris (el azar), mejor separa el modelo ambas clases. "
     "AUC=0.814 es un resultado sólido para un problema con señal débil y clase minoritaria tan escasa. "
     "<strong>Haz zoom</strong> arrastrando sobre una zona de la curva para ver el detalle."),
    (fig3, "3 · Dónde acierta y dónde falla en test",
     "De 123 clientes que sí eran mal pagador en el conjunto de prueba, el modelo detectó 45 (recall 37%) y dejó pasar 78 sin alertar. "
     "De los 7,169 buenos pagadores, marcó 201 como falso riesgo. "
     "Esta es la foto honesta del costo del modelo: prioriza no dejar pasar tantos malos pagadores como sea posible, a cambio de generar falsas alarmas. "
     "<strong>Pasa el mouse</strong> sobre cada celda para ver el valor."),
    (fig4, "4 · Qué variables pesan más en la decisión",
     "Random Forest calcula cuánto reduce cada variable la incertidumbre del modelo al dividir los datos. "
     "Edad, ingreso total y antigüedad laboral concentran casi la mitad de la señal predictiva — consistente con cómo un analista de crédito evaluaría el riesgo manualmente. "
     "<strong>Haz scroll/zoom</strong> para comparar el peso relativo de cada variable."),
]
html_partes = []
for fig, titulo, descripcion in graficas:
    bloque = pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        default_width="100%", default_height="100%",
        config={"displayModeBar": True, "displaylogo": False, "responsive": True},
    )
    html_partes.append(f'''
  <section class="panel">
    <h2>{titulo}</h2>
    <p class="descripcion">{descripcion}</p>
    <div class="plot-wrap" style="aspect-ratio:980/{fig._alto_ref};">{bloque}</div>
  </section>''')

pagina = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scoring Crediticio — Dashboard interactivo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:{BG}; color:#eee; font-family: 'Segoe UI', system-ui, sans-serif; max-width: 980px; margin: 0 auto; padding: 30px 20px 60px; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.15rem; margin: 0 0 10px; color: #fff; }}
  p.intro {{ color:#999; margin-top: 0; }}
  .descripcion {{ color:#bbb; font-size: 0.95rem; margin-bottom: 14px; max-width: 760px; }}
  .descripcion code {{ background:#2a2a2a; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
  a {{ color:{AZUL}; }}
  .kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 24px 0 36px; }}
  .kpi {{ background:#262626; border-radius: 10px; padding: 16px 18px; }}
  .kpi-valor {{ font-size: 1.8rem; font-weight: 700; }}
  .kpi-label {{ font-size: 0.85rem; color:#ccc; margin-top: 2px; font-weight: 600; }}
  .kpi-nota {{ font-size: 0.78rem; color:#888; margin-top: 6px; }}
  .panel {{ background:#242424; border-radius: 12px; padding: 20px 24px 8px; margin-bottom: 28px; }}
  footer {{ color:#666; font-size: 0.85rem; margin-top: 20px; }}
  .plot-wrap {{ width: 100%; position: relative; }}
  .plot-wrap .plotly-graph-div {{ width: 100% !important; height: 100% !important; position: absolute; inset: 0; }}
  @media (max-width: 640px) {{
    .plot-wrap {{ aspect-ratio: 3 / 4 !important; }}
  }}
</style>
</head>
<body>
<h1>Scoring Crediticio — Predicción de Riesgo de Impago</h1>
<p class="intro">Dashboard interactivo del análisis. Repo completo: <a href="https://github.com/cjhirashi/ds-scoring-credito">github.com/cjhirashi/ds-scoring-credito</a> · <a href="https://github.com/cjhirashi/ds-scoring-credito/blob/main/notebooks/analisis_scoring_crediticio.ipynb">Notebook</a></p>
<div class="kpis">{kpi_html}</div>
{''.join(html_partes)}
<footer>Generado con Random Forest + scikit-learn, visualizado con Plotly. Dataset: Kaggle <code>rikdifos/credit-card-approval-prediction</code>.</footer>
</body>
</html>"""

with open(f"{RUTA_DOCS}/index.html", "w", encoding="utf-8") as f:
    f.write(pagina)

print("página interactiva generada en docs/index.html")
