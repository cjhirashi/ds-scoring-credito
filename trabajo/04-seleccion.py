import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

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

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("prep", preprocesador),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
])

param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [6, 10, None],
    "clf__min_samples_leaf": [1, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
busqueda = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
busqueda.fit(X_train, y_train)

print("mejores hiperparámetros:", busqueda.best_params_)
print("roc_auc cv (train):", round(busqueda.best_score_, 4))

mejor = busqueda.best_estimator_
proba_test = mejor.predict_proba(X_test)[:, 1]
pred_test = mejor.predict(X_test)

print("\n=== evaluación en test (nunca visto) ===")
print("roc_auc:", round(roc_auc_score(y_test, proba_test), 4))
print("f1:", round(f1_score(y_test, pred_test), 4))
print("precision:", round(precision_score(y_test, pred_test, zero_division=0), 4))
print("recall:", round(recall_score(y_test, pred_test), 4))
print(classification_report(y_test, pred_test, target_names=["bueno", "malo"]))

resumen = pd.DataFrame([{
    "mejores_params": str(busqueda.best_params_),
    "roc_auc_cv_train": round(busqueda.best_score_, 4),
    "roc_auc_test": round(roc_auc_score(y_test, proba_test), 4),
    "f1_test": round(f1_score(y_test, pred_test), 4),
    "precision_test": round(precision_score(y_test, pred_test, zero_division=0), 4),
    "recall_test": round(recall_score(y_test, pred_test), 4),
}])
resumen.to_csv(f"{RUTA_RES}/modelo_final.csv", index=False)

# Modelo de producción: mismos hiperparámetros, reentrenado con TODOS los datos
# (train+test) para aprovechar toda la señal disponible una vez ya evaluado el
# desempeño honesto arriba. Las métricas reportadas siguen siendo las de test.
modelo_produccion = busqueda.best_estimator_
modelo_produccion.fit(X, y)
joblib.dump(modelo_produccion, f"{RUTA_RES}/modelo_final.joblib")

with open(f"{RUTA_RES}/esquema_entrada.json", "w") as f:
    json.dump({"num_cols": num_cols, "cat_cols": cat_cols}, f, indent=2)

print("\nmodelo de producción guardado en resultados/modelo_final.joblib")
