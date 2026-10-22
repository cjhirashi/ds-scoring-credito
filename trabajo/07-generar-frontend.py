import json
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
AZUL, ROJO, VERDE, NARANJA = "#4C9AFF", "#FF6B6B", "#5FD3A3", "#F5A623"

RANDOM_STATE = 42
RUTA_RES = "../resultados"
RUTA_APP_STATIC = "../app/static"

df = pd.read_csv(f"{RUTA_RES}/dataset_limpio.csv")
y = df["es_malo"]
X = df.drop(columns=["ID", "es_malo"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

def layout_base(fig, titulo, height=440):
    fig.update_layout(title=titulo, paper_bgcolor=BG, plot_bgcolor=BG,
                       font=dict(size=13), margin=dict(l=55, r=25, t=50, b=45), height=height)
    return fig

# Gráficas (mismo contenido que 06-graficas-interactivas.py)
conteo = y.value_counts().rename({0: "Buen pagador", 1: "Mal pagador"})
fig1 = layout_base(go.Figure(go.Bar(
    x=conteo.index, y=conteo.values, marker_color=[AZUL, ROJO],
    text=[f"{v:,} ({v/len(y)*100:.1f}%)" for v in conteo.values], textposition="outside",
)), f"Distribución de la etiqueta (n={len(y):,})")
fig1.update_layout(yaxis_title="Clientes")

preprocesador = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
modelo = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=1,
                                 class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
pipe.fit(X_train, y_train)
proba_test = pipe.predict_proba(X_test)[:, 1]
pred_test = pipe.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, proba_test)
auc = roc_auc_score(y_test, proba_test)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={auc:.3f})",
                           line=dict(color=AZUL, width=3), fill="tozeroy", fillcolor="rgba(76,154,255,0.15)"))
fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC=0.5)", line=dict(color="gray", dash="dash")))
fig2 = layout_base(fig2, "Curva ROC — modelo final en test")
fig2.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

cm = confusion_matrix(y_test, pred_test)
etiquetas = ["Buen pagador", "Mal pagador"]
fig3 = go.Figure(go.Heatmap(z=cm, x=etiquetas, y=etiquetas, colorscale="Teal",
                             text=cm, texttemplate="%{text}", textfont=dict(size=18), showscale=False))
fig3 = layout_base(fig3, "Matriz de confusión — test")
fig3.update_layout(xaxis_title="Predicción", yaxis_title="Real", yaxis=dict(autorange="reversed"))

nombres_cat = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
nombres = list(num_cols) + list(nombres_cat)
importancias = pipe.named_steps["clf"].feature_importances_
top = pd.Series(importancias, index=nombres).sort_values(ascending=True).tail(15)
fig4 = layout_base(go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                     marker=dict(color=top.values, colorscale="Teal"))),
                    "Top 15 variables más importantes", height=520)
fig4.update_layout(xaxis_title="Importancia")

graficas = [
    (fig1, "1 · El problema: clases muy desbalanceadas",
     "Solo 1.7% de los 36,457 clientes con historial cruzado son mal pagador. Esto obliga a usar "
     "<code>class_weight=\"balanced\"</code> y a medir con ROC-AUC/recall en vez de accuracy."),
    (fig2, "2 · Qué tan bien discrimina el modelo",
     "AUC=0.814: entre más se aleje la curva azul de la diagonal gris (el azar), mejor separa el modelo ambas clases."),
    (fig3, "3 · Dónde acierta y dónde falla en test",
     "De 123 mal pagador reales en test, detecta 45 (recall 37%); de 7,169 buenos pagadores, marca 201 como falso riesgo."),
    (fig4, "4 · Qué variables pesan más en la decisión",
     "Edad, ingreso total y antigüedad laboral concentran casi la mitad de la señal predictiva."),
]
html_graficas = []
for fig, titulo, descripcion in graficas:
    bloque = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": True, "displaylogo": False})
    html_graficas.append(f'<section class="panel"><h2>{titulo}</h2><p class="descripcion">{descripcion}</p>{bloque}</section>')

kpis = [
    ("ROC-AUC (test)", f"{auc:.3f}", "Capacidad de separar mal pagador de buen pagador. 0.5=azar, 1.0=perfecto.", AZUL),
    ("Desbalance", f"{(y.mean()*100):.1f}%", "Proporción de mal pagador sobre el total con historial.", ROJO),
    ("Recall", "36.6%", "De cada 10 malos pagadores reales, detecta ~4.", VERDE),
    ("Precision", "18.3%", "De cada 10 alertas, ~2 son realmente mal pagador.", NARANJA),
]
kpi_html = "".join(f'''<div class="kpi" style="border-top:3px solid {color}">
  <div class="kpi-valor" style="color:{color}">{valor}</div>
  <div class="kpi-label">{label}</div><div class="kpi-nota">{nota}</div></div>''' for label, valor, nota, color in kpis)

# --- Formulario de predicción ---
with open(f"{RUTA_RES}/valores_categoricos.json") as f:
    valores_cat = json.load(f)

ETIQUETAS_CAMPO = {
    "CODE_GENDER": "Género", "FLAG_OWN_CAR": "¿Tiene auto?", "FLAG_OWN_REALTY": "¿Tiene inmueble?",
    "NAME_INCOME_TYPE": "Tipo de ingreso", "NAME_EDUCATION_TYPE": "Educación",
    "NAME_FAMILY_STATUS": "Estado civil", "NAME_HOUSING_TYPE": "Vivienda", "OCCUPATION_TYPE": "Ocupación",
}
selects = []
for campo, opciones in valores_cat.items():
    opts = "".join(f'<option value="{o}">{o}</option>' for o in opciones)
    selects.append(f'<label>{ETIQUETAS_CAMPO[campo]}<select name="{campo}">{opts}</select></label>')

form_html = f'''
<section class="panel">
  <h2>5 · Prueba el modelo con un registro nuevo</h2>
  <p class="descripcion">Llena los datos de un solicitante y el modelo calcula su probabilidad de convertirse en mal pagador.</p>
  <form id="form-prediccion" class="form-grid">
    {''.join(selects)}
    <label>Edad (años)<input type="number" name="EDAD_ANIOS" min="18" max="100" value="35" required></label>
    <label>Años de empleo<input type="number" name="ANIOS_EMPLEO" min="0" max="60" value="5" step="0.1" required></label>
    <label>Ingreso total anual<input type="number" name="AMT_INCOME_TOTAL" min="1" value="200000" required></label>
    <label>Número de hijos<input type="number" name="CNT_CHILDREN" min="0" max="20" value="0" required></label>
    <label>Miembros de familia<input type="number" name="CNT_FAM_MEMBERS" min="1" max="20" value="2" required></label>
    <label class="checkbox"><input type="checkbox" name="DESEMPLEADO"> Actualmente desempleado/pensionado</label>
    <label class="checkbox"><input type="checkbox" name="FLAG_WORK_PHONE"> Tiene teléfono de trabajo</label>
    <label class="checkbox"><input type="checkbox" name="FLAG_PHONE"> Tiene teléfono</label>
    <label class="checkbox"><input type="checkbox" name="FLAG_EMAIL"> Tiene email</label>
    <button type="submit">Calcular riesgo</button>
  </form>
  <div id="resultado-prediccion"></div>
</section>
'''

pagina = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scoring Crediticio — Dashboard</title>
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
  .form-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; padding-bottom: 16px; }}
  .form-grid label {{ display:flex; flex-direction:column; font-size:0.85rem; color:#ccc; gap:4px; }}
  .form-grid label.checkbox {{ flex-direction:row; align-items:center; gap:8px; }}
  .form-grid select, .form-grid input[type=number] {{ background:#1a1a1a; border:1px solid #444; color:#eee; border-radius:6px; padding:8px; }}
  button {{ grid-column: 1 / -1; background:{AZUL}; color:#111; font-weight:700; border:none; border-radius:8px; padding:12px; cursor:pointer; font-size:1rem; }}
  button:hover {{ opacity:0.9; }}
  #resultado-prediccion {{ padding: 16px 0; font-size: 1rem; }}
  .resultado-ok {{ color:{VERDE}; font-weight:700; }}
  .resultado-riesgo {{ color:{ROJO}; font-weight:700; }}
  footer {{ color:#666; font-size: 0.85rem; margin-top: 20px; }}
</style>
</head>
<body>
<h1>Scoring Crediticio — Predicción de Riesgo de Impago</h1>
<p class="intro">Dashboard interactivo + modelo en vivo. Repo: <a href="https://github.com/cjhirashi/ds-scoring-credito">github.com/cjhirashi/ds-scoring-credito</a></p>
<div class="kpis">{kpi_html}</div>
{''.join(html_graficas)}
{form_html}
<footer>Random Forest + scikit-learn, servido con FastAPI, visualizado con Plotly. Dataset: Kaggle <code>rikdifos/credit-card-approval-prediction</code>.</footer>
<script>
document.getElementById('form-prediccion').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const form = e.target;
  const data = {{}};
  new FormData(form).forEach((v, k) => data[k] = v);
  ['FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL','DESEMPLEADO'].forEach(k => {{
    data[k] = form.querySelector(`[name=${{k}}]`)?.type === 'checkbox' ? (form.querySelector(`[name=${{k}}]`).checked ? 1 : 0) : (data[k] ?? 0);
  }});
  data.FLAG_MOBIL = 1;
  ['CNT_CHILDREN','CNT_FAM_MEMBERS','EDAD_ANIOS','ANIOS_EMPLEO','AMT_INCOME_TOTAL'].forEach(k => data[k] = Number(data[k]));
  const resDiv = document.getElementById('resultado-prediccion');
  resDiv.textContent = 'Calculando...';
  try {{
    const resp = await fetch('/predict', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(data) }});
    if (!resp.ok) {{ const err = await resp.json(); throw new Error(JSON.stringify(err)); }}
    const r = await resp.json();
    const pct = (r.probabilidad_mal_pagador * 100).toFixed(1);
    resDiv.innerHTML = `<span class="${{r.es_riesgo ? 'resultado-riesgo' : 'resultado-ok'}}">
      Probabilidad de mal pagador: ${{pct}}% — ${{r.es_riesgo ? 'RIESGO' : 'SIN RIESGO SIGNIFICATIVO'}}</span><br>
      <span style="color:#999;font-size:0.85rem">${{r.nota}}</span>`;
  }} catch (err) {{
    resDiv.textContent = 'Error: ' + err.message;
  }}
}});
</script>
</body>
</html>"""

with open(f"{RUTA_APP_STATIC}/index.html", "w", encoding="utf-8") as f:
    f.write(pagina)

print("frontend generado en app/static/index.html")
