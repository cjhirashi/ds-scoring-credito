from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .schemas import SolicitudCredito, ResultadoPrediccion

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_final.joblib"
UMBRAL = 0.5

app = FastAPI(title="Scoring Crediticio API", version="1.0")
modelo = joblib.load(MODELO_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ResultadoPrediccion)
def predict(solicitud: SolicitudCredito):
    fila = pd.DataFrame([solicitud.model_dump()])
    proba = float(modelo.predict_proba(fila)[0, 1])
    es_riesgo = proba >= UMBRAL
    nota = (
        "El modelo tiene precisión de solo 18% en este umbral: útil como filtro "
        "de revisión manual, no como decisión automática de rechazo."
        if es_riesgo
        else "Sin señales fuertes de riesgo según el modelo."
    )
    return ResultadoPrediccion(
        probabilidad_mal_pagador=round(proba, 4),
        es_riesgo=es_riesgo,
        umbral_usado=UMBRAL,
        nota=nota,
    )


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
