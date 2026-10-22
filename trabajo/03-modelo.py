import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

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

modelos = {
    "baseline_dummy": DummyClassifier(strategy="most_frequent"),
    "regresion_logistica": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "arbol_decision": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
}

resultados = []
for nombre, modelo in modelos.items():
    pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
    pipe.fit(X_train, y_train)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    else:
        auc = float("nan")
    pred = pipe.predict(X_test)
    resultados.append({
        "modelo": nombre,
        "roc_auc": round(auc, 4),
        "f1": round(f1_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
    })

tabla = pd.DataFrame(resultados).sort_values("roc_auc", ascending=False)
print(tabla.to_string(index=False))
tabla.to_csv(f"{RUTA_RES}/comparacion_modelos.csv", index=False)
