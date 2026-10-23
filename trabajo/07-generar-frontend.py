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
    # height queda como "alto de referencia" (usado por el contenedor CSS vía
    # aspect-ratio, no fijado aquí) — autosize:True deja que Plotly llene ese
    # contenedor en vez de imponer un ancho/alto fijo que se ve mal en móvil.
    fig.update_layout(title=titulo, paper_bgcolor=BG, plot_bgcolor=BG,
                       font=dict(size=13), margin=dict(l=50, r=20, t=50, b=40),
                       autosize=True,
                       xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    fig._alto_ref = height
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
fig2 = layout_base(fig2, "Curva ROC — modelo final en test", height=620)
fig2.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

cm = confusion_matrix(y_test, pred_test)
etiquetas = ["Buen pagador", "Mal pagador"]
fig3 = go.Figure(go.Heatmap(z=cm, x=etiquetas, y=etiquetas, colorscale="Teal",
                             text=cm, texttemplate="%{text}", textfont=dict(size=18), showscale=False))
fig3 = layout_base(fig3, "Matriz de confusión — test", height=560)
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
    ancho_ref = 980
    bloque = pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        default_width="100%", default_height="100%",
        config={"displayModeBar": True, "displaylogo": False, "responsive": True},
    )
    html_graficas.append(
        f'<section class="panel card"><h2>{titulo}</h2><p class="descripcion">{descripcion}</p>'
        f'<div class="plot-wrap" style="aspect-ratio:{ancho_ref}/{fig._alto_ref};">{bloque}</div></section>'
    )

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

  /* Gráficas Plotly — responsivas: el wrapper fija la proporción con
     aspect-ratio, Plotly llena ese contenedor (config responsive:true)
     en vez de imponer un ancho/alto fijo en px que se ve mal en móvil. */
  .plot-wrap {{ width: 100%; position: relative; }}
  .plot-wrap .plotly-graph-div {{ width: 100% !important; height: 100% !important; position: absolute; inset: 0; }}

  @media (max-width: 640px) {{
    .wrap {{ padding: 0 14px 60px; }}
    .hero {{ padding: 48px 0 28px; }}
    .panel {{ padding: 18px 16px 10px; }}
    .modebar-container {{ display: none !important; }}
    /* En móvil la proporción de escritorio (ancho_ref/alto_ref, ~980/440)
       deja el contenedor demasiado bajo y el título choca con los ejes.
       Se fuerza una relación más alta (!important vence al aspect-ratio
       inline, ya que un !important de hoja de estilos supera a un estilo
       inline sin !important). */
    .plot-wrap {{ aspect-ratio: 3 / 4 !important; }}
  }}

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

  /* Footer — misma estructura que cjhirashi-career-portfolio (Footer.tsx): copyright + iconos sociales */
  footer {{
    color: var(--text-secondary); margin-top: 48px; border-top: 1px solid var(--border-glass);
  }}
  .footer-row {{
    padding: 28px 0 8px;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px;
  }}
  .footer-row p {{ font-size: 0.85rem; margin: 0; }}
  .footer-social {{ display: flex; align-items: center; gap: 20px; }}
  .footer-social a {{ color: var(--text-secondary); display: inline-flex; transition: color 150ms ease; }}
  .footer-social a:hover {{ color: var(--primary-color); text-decoration: none; }}
  .footer-back {{ font-size: 0.85rem; color: var(--text-secondary) !important; }}
  .footer-back:hover {{ color: var(--primary-color) !important; }}
  .footer-stack {{ padding: 0 0 24px; font-size: 0.78rem; color: var(--text-muted); }}
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
    <div class="footer-row">
      <div>
        <p>© 2026 Carlos A. Jiménez Hirashi</p>
        <a class="footer-back" href="https://cjhirashi.com/projects/prj-47" target="_blank" rel="noopener noreferrer">← Volver al proyecto en el portafolio</a>
      </div>
      <div class="footer-social">
        <a href="https://github.com/cjhirashi" target="_blank" rel="noopener noreferrer" aria-label="GitHub" title="GitHub">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56 0-.27-.01-1-.02-1.97-3.2.7-3.88-1.54-3.88-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.33.95.1-.74.4-1.24.72-1.53-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.29 1.18-3.09-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.24 2.76.12 3.05.74.8 1.18 1.83 1.18 3.09 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.07.78 2.15 0 1.56-.01 2.81-.01 3.19 0 .31.21.67.8.56A10.52 10.52 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z"/></svg>
        </a>
        <a href="https://www.linkedin.com/in/cjhirashi" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" title="LinkedIn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.34V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.07 2.07 0 1 1 0-4.13 2.07 2.07 0 0 1 0 4.13ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.8 0 0 .78 0 1.75v20.5C0 23.22.8 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.75V1.75C24 .78 23.2 0 22.22 0Z"/></svg>
        </a>
        <a href="mailto:carlos@cjhirashi.com" aria-label="Email" title="Email">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 7L2 7"/></svg>
        </a>
        <a href="https://wa.me/525513710160" target="_blank" rel="noopener noreferrer" aria-label="WhatsApp" title="WhatsApp">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.868-2.03-.967-.273-.099-.47-.148-.669.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z"/><path d="M12.041 2C6.496 2 2 6.476 2 11.994c0 1.993.588 3.85 1.6 5.411L2 22l4.71-1.555a10.09 10.09 0 0 0 5.331 1.521h.004c5.545 0 10.04-4.476 10.042-9.994 0-2.67-1.04-5.179-2.93-7.064A10.013 10.013 0 0 0 12.041 2Zm0 18.168h-.003a8.157 8.157 0 0 1-4.168-1.14l-.299-.177-3.096 1.023.84-3.5-.199-.33a8.13 8.13 0 0 1-1.262-4.37c0-4.502 3.68-8.166 8.164-8.166a8.14 8.14 0 0 1 5.783 2.4 8.1 8.1 0 0 1 2.39 5.785c-.002 4.502-3.68 8.475-8.15 8.475Z"/></svg>
        </a>
      </div>
    </div>
    <div class="footer-stack mono">Random Forest + scikit-learn · FastAPI · Plotly · Dataset: Kaggle rikdifos/credit-card-approval-prediction</div>
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
