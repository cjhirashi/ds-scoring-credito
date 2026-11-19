import ast
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

RANDOM_STATE = 42
R = "../resultados"

params = ast.literal_eval(pd.read_csv(f"{R}/modelo_final.csv").mejores_params[0])
df = pd.read_csv(f"{R}/dataset_limpio.csv")
y = df["es_malo"]
X = df.drop(columns=["ID", "es_malo"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pipe = Pipeline([
    ("prep", ColumnTransformer([("n", StandardScaler(), num_cols),
                                ("c", OneHotEncoder(handle_unknown="ignore"), cat_cols)])),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
]).set_params(**params)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
pipe.fit(X_tr, y_tr)
p_te = pipe.predict_proba(X_te)[:, 1]

print(f"AUC test {roc_auc_score(y_te, p_te):.3f} | PR-AUC test {average_precision_score(y_te, p_te):.3f} "
      f"(azar = {y_te.mean():.3f}) | malos test {int(y_te.sum())} de {len(y_te)}")


def metricas(y_true, p, t):
    pred = p >= t
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum())
    return tp, fp, fn


def costo(y_true, p, t, ratio):
    tp, fp, fn = metricas(y_true, p, t)
    return ratio * fn + fp


umbrales = np.round(np.arange(0.01, 0.80, 0.01), 2)
resultado = {}
for ratio in [1, 5, 10, 20]:
    costos = [costo(y_tr, p_oof, t, ratio) for t in umbrales]
    t_opt = float(umbrales[int(np.argmin(costos))])
    tp, fp, fn = metricas(y_te, p_te, t_opt)
    base_tp, base_fp, base_fn = metricas(y_te, p_te, 0.5)
    resultado[ratio] = {
        "umbral": t_opt,
        "recall_test": round(tp / (tp + fn), 3),
        "precision_test": round(tp / max(tp + fp, 1), 3),
        "marcados_pct": round((tp + fp) / len(y_te) * 100, 1),
        "costo_test": ratio * fn + fp,
        "costo_test_umbral_05": ratio * base_fn + base_fp,
        "costo_sin_modelo_aprobar_todos": ratio * int(y_te.sum()),
    }
    print(f"\nCosto FN:FP = {ratio}:1 -> umbral {t_opt} (elegido en train, fuera de muestra)")
    for k, v in resultado[ratio].items():
        print(f"  {k}: {v}")

with open(f"{R}/analisis_umbral.json", "w") as f:
    json.dump({str(k): v for k, v in resultado.items()}, f, indent=2)

# figura
plt.style.use("dark_background")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
prec, rec, thr = precision_recall_curve(y_te, p_te)
ax[0].plot(rec, prec, color="#4cc9f0", lw=2)
ax[0].axhline(y_te.mean(), color="gray", ls="--", label=f"azar ({y_te.mean():.3f})")
for t, c in [(0.5, "#f72585"), (resultado[10]["umbral"], "#b8f2e6")]:
    tp, fp, fn = metricas(y_te, p_te, t)
    ax[0].scatter(tp / (tp + fn), tp / max(tp + fp, 1), color=c, s=90, zorder=5,
                  label=f"umbral {t:.2f}")
ax[0].set_xlabel("Recall (malos detectados)")
ax[0].set_ylabel("Precisión")
ax[0].set_title("Curva precisión–recall (test)")
ax[0].legend()
for ratio, c in [(5, "#ffd166"), (10, "#4cc9f0"), (20, "#f72585")]:
    ax[1].plot(umbrales, [costo(y_tr, p_oof, t, ratio) for t in umbrales], color=c, label=f"FN cuesta {ratio}×FP")
    ax[1].axvline(resultado[ratio]["umbral"], color=c, ls=":", alpha=0.6)
ax[1].set_xlabel("Umbral de decisión")
ax[1].set_ylabel("Costo total (unidades de FP, train fuera de muestra)")
ax[1].set_title("Umbral que minimiza el costo")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{R}/fig_05_umbral.png", dpi=130)
print("\nfigura guardada")
