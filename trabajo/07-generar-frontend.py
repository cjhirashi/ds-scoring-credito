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
BG = "#0c1822"
AZUL, ROJO, VERDE, NARANJA = "#06b6d4", "#FF6B6B", "#5FD3A3", "#F5A623"

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
                           line=dict(color=AZUL, width=3), fill="tozeroy", fillcolor="rgba(6,182,212,0.15)"))
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
    html_graficas.append(f'<section class="panel card"><h2>{titulo}</h2><p class="descripcion">{descripcion}</p>{bloque}</section>')

kpis = [
    ("ROC-AUC (test)", f"{auc:.3f}", "Capacidad de separar mal pagador de buen pagador. 0.5=azar, 1.0=perfecto.", AZUL),
    ("Desbalance", f"{(y.mean()*100):.1f}%", "Proporción de mal pagador sobre el total con historial.", ROJO),
    ("Recall", "36.6%", "De cada 10 malos pagadores reales, detecta ~4.", VERDE),
    ("Precision", "18.3%", "De cada 10 alertas, ~2 son realmente mal pagador.", NARANJA),
]
kpi_html = "".join(f'''<div class="stat-card" style="--kpi-color:{color}">
  <div class="stat-card-head"><span>{label}</span></div>
  <h4>{valor}</h4>
  <div class="kpi-nota">{nota}</div></div>''' for label, valor, nota, color in kpis)

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
<section class="panel card">
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
<title>Scoring Crediticio — Carlos Jiménez Hirashi</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  /*
   * Tokens "Glass Steel" copiados literal de cjhirashi-career-portfolio
   * (src/index.css, tema [data-theme="dark"]) — no reconstruidos de memoria.
   */
  :root {{
    --bg-primary: #071018;
    --bg-secondary: rgba(16, 30, 42, 0.6);
    --bg-glass: rgba(12, 24, 34, 0.7);
    --bg-card: rgba(26, 48, 64, 0.5);
    --text-primary: #f8fafc;
    --text-secondary: #a3b6c0;
    --text-muted: #6b808a;
    --border-glass: rgba(255, 255, 255, 0.09);
    --border-glass-hover: rgba(34, 211, 238, 0.5);
    --primary-color: #06b6d4;
    --primary-hover: #0891b2;
    --primary-light: rgba(34, 211, 238, 0.16);
    --primary-glow: rgba(34, 211, 238, 0.35);
    --secondary-color: #94a3b8;
    --secondary-hover: #64748b;
    --secondary-light: rgba(148, 163, 184, 0.16);
    --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    --shadow-glow: 0 0 30px var(--primary-glow);
    --radius-sm: 12px; --radius-md: 16px; --radius-lg: 24px;
    --ambient-cyan: rgba(34, 211, 238, 0.2);
    --ambient-steel: rgba(148, 163, 184, 0.18);
    --error-bg: rgba(239, 68, 68, 0.15); --error-text: #f87171; --error-border: rgba(239, 68, 68, 0.3);
    --warning-bg: rgba(245, 158, 11, 0.15); --warning-text: #fbbf24; --warning-border: rgba(245, 158, 11, 0.3);
    --success-bg: rgba(16, 185, 129, 0.15); --success-text: #34d399; --success-border: rgba(16, 185, 129, 0.3);
    --rojo: {ROJO}; --verde: {VERDE}; --naranja: {NARANJA};
    --font: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    color: var(--text-primary); font-family: var(--font); margin: 0; line-height: 1.6;
    -webkit-font-smoothing: antialiased; overflow-x: hidden;
    background-color: var(--bg-primary);
    background-image:
      radial-gradient(circle at 10% 20%, var(--ambient-cyan) 0%, transparent 45%),
      radial-gradient(circle at 90% 80%, var(--ambient-steel) 0%, transparent 45%);
    background-attachment: fixed; background-repeat: no-repeat; background-size: cover;
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 0 24px 80px; position: relative; }}
  a {{ color: var(--primary-color); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font-family: var(--mono); background: var(--bg-card); padding: 2px 6px; border-radius: 5px; font-size: 0.85em; border: 1px solid var(--border-glass); }}

  /* .card — idéntica a cjhirashi-career-portfolio */
  .card {{
    background-color: var(--bg-card); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass); border-radius: var(--radius-md); box-shadow: var(--shadow-glass);
    transition: transform 250ms ease, border-color 250ms ease, box-shadow 250ms ease;
  }}
  .card:hover {{ transform: translateY(-4px); border-color: var(--border-glass-hover); }}

  /* .badge + variantes — idénticas */
  .badge {{
    display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.8rem;
    border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    backdrop-filter: blur(8px); border: 1px solid transparent; margin-bottom: 20px;
  }}
  .badge-primary {{ background-color: var(--primary-light); color: var(--primary-color); border-color: rgba(34, 211, 238, 0.3); }}
  .badge-secondary {{ background-color: var(--secondary-light); color: var(--secondary-color); border-color: rgba(148, 163, 184, 0.3); }}
  .badge-success {{ background-color: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }}
  .badge-warning {{ background-color: var(--warning-bg); color: var(--warning-text); border-color: var(--warning-border); }}
  .badge-error {{ background-color: var(--error-bg); color: var(--error-text); border-color: var(--error-border); }}

  /* .btn / .btn-secondary — idénticas */
  .btn {{
    display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 11px 20px;
    background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)); color: #fff;
    border: none; border-radius: var(--radius-sm); box-shadow: 0 4px 20px var(--primary-glow);
    font-weight: 600; font-size: 0.92rem; cursor: pointer;
    transition: transform 200ms ease, box-shadow 200ms ease, opacity 200ms ease;
  }}
  .btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 28px var(--primary-glow); text-decoration: none; }}
  .btn-secondary {{
    display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 11px 20px;
    background-color: var(--bg-glass); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    color: var(--text-primary); border: 1px solid var(--border-glass); border-radius: var(--radius-sm);
    font-weight: 600; font-size: 0.92rem;
    transition: transform 200ms ease, border-color 200ms ease;
  }}
  .btn-secondary:hover {{ transform: translateY(-2px); border-color: var(--border-glass-hover); color: var(--primary-color); text-decoration: none; }}

  /* Hero */
  .hero {{ padding: 72px 0 40px; text-align: left; }}
  .hero h1 {{
    font-size: clamp(2rem, 4.5vw, 2.9rem); font-weight: 800; letter-spacing: -0.02em;
    margin: 0 0 16px; line-height: 1.12; color: var(--text-primary);
  }}
  .hero p.lead {{ font-size: 1.1rem; color: var(--text-secondary); max-width: 640px; margin: 0 0 28px; }}
  .hero .links {{ display: flex; gap: 12px; flex-wrap: wrap; }}

  h2 {{ font-size: 1.2rem; font-weight: 700; margin: 0 0 8px; color: var(--text-primary); letter-spacing: -0.01em; }}

  /* KPIs — patrón .stat-card idéntico al dashboard.html de referencia */
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 8px 0 48px; }}
  .stat-card {{
    background-color: var(--bg-card); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass); border-radius: var(--radius-md); padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-glass); display: flex; flex-direction: column; gap: 0.5rem;
  }}
  .stat-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }}
  .stat-card-head > span {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; }}
  .stat-card h4 {{ font-size: 1.875rem; font-weight: 700; font-family: var(--mono); margin: 0; color: var(--kpi-color, var(--text-primary)); }}
  .kpi-nota {{ font-size: 0.78rem; color: var(--text-muted); line-height: 1.4; }}

  /* Panels */
  .panel {{ padding: 28px 28px 12px; margin-bottom: 24px; }}
  .descripcion {{ color: var(--text-secondary); font-size: 0.95rem; margin: 0 0 18px; max-width: 760px; }}

  /* Formulario — mismo tratamiento que .input-field */
  .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; padding-bottom: 20px; }}
  .form-grid label {{ display: flex; flex-direction: column; font-size: 0.82rem; color: var(--text-secondary); gap: 6px; font-weight: 500; }}
  .form-grid label.checkbox {{ flex-direction: row; align-items: center; gap: 10px; }}
  .form-grid select, .form-grid input[type=number] {{
    background-color: var(--bg-card); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass); color: var(--text-primary);
    border-radius: var(--radius-sm); padding: 10px 12px; font-family: var(--font); font-size: 0.92rem;
    transition: border-color 200ms ease, box-shadow 200ms ease;
  }}
  .form-grid select:focus, .form-grid input:focus {{
    outline: none; border-color: var(--border-glass-hover); box-shadow: 0 0 0 2px var(--primary-glow);
  }}
  .form-grid input[type=checkbox] {{ width: 16px; height: 16px; accent-color: var(--primary-color); }}
  button[type=submit] {{
    grid-column: 1 / -1; background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)); color: #fff;
    font-weight: 700; border: none; border-radius: var(--radius-sm); padding: 14px; cursor: pointer;
    font-size: 0.98rem; margin-top: 4px; transition: all .15s ease;
  }}
  button[type=submit]:hover {{ background: var(--primary-hover); box-shadow: 0 6px 24px var(--primary-glow); transform: translateY(-1px); }}
  #resultado-prediccion {{ padding: 18px 0 20px; font-size: 1rem; min-height: 1.2em; }}
  .resultado-ok {{ color: var(--verde); font-weight: 700; }}
  .resultado-riesgo {{ color: var(--rojo); font-weight: 700; }}

  .mono {{ font-family: var(--mono); }}

  footer {{
    color: var(--text-muted); font-size: 0.85rem; margin-top: 40px; padding-top: 24px;
    border-top: 1px solid var(--border-glass); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  }}
  footer .mono {{ font-family: var(--mono); }}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
      <span class="badge badge-success">Modelo en producción</span>
      <span class="badge badge-primary">API en vivo</span>
    </div>
    <h1>Scoring Crediticio: predicción de riesgo de impago</h1>
    <p class="lead">Random Forest entrenado sobre 36,457 solicitudes de tarjeta de crédito para anticipar qué clientes se convertirán en mal pagador, con ROC-AUC 0.814 en datos nunca vistos. Explora el análisis completo abajo y prueba el modelo con tus propios datos.</p>
    <div class="links">
      <a class="btn" href="#predecir">Probar el modelo →</a>
      <a class="btn-secondary" href="https://github.com/cjhirashi/ds-scoring-credito" target="_blank" rel="noopener noreferrer">Ver repositorio</a>
      <a class="btn-secondary" href="https://github.com/cjhirashi/ds-scoring-credito/blob/main/notebooks/analisis_scoring_crediticio.ipynb" target="_blank" rel="noopener noreferrer">Ver notebook</a>
    </div>
  </section>

  <div class="kpis">{kpi_html}</div>
  {''.join(html_graficas)}
  <div id="predecir">{form_html}</div>

  <footer>
    <span>Carlos Jiménez Hirashi — Data Scientist &amp; Intelligent Automation Architect</span>
    <span class="mono">Random Forest + scikit-learn · FastAPI · Plotly · Dataset: Kaggle rikdifos/credit-card-approval-prediction</span>
  </footer>
</div>
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
