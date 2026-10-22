from typing import Literal
from pydantic import BaseModel, Field


class SolicitudCredito(BaseModel):
    CODE_GENDER: Literal["F", "M"]
    FLAG_OWN_CAR: Literal["N", "Y"]
    FLAG_OWN_REALTY: Literal["N", "Y"]
    CNT_CHILDREN: int = Field(ge=0, le=20)
    AMT_INCOME_TOTAL: float = Field(gt=0)
    NAME_INCOME_TYPE: Literal[
        "Commercial associate", "Pensioner", "State servant", "Student", "Working"
    ]
    NAME_EDUCATION_TYPE: Literal[
        "Academic degree", "Higher education", "Incomplete higher",
        "Lower secondary", "Secondary / secondary special",
    ]
    NAME_FAMILY_STATUS: Literal[
        "Civil marriage", "Married", "Separated", "Single / not married", "Widow"
    ]
    NAME_HOUSING_TYPE: Literal[
        "Co-op apartment", "House / apartment", "Municipal apartment",
        "Office apartment", "Rented apartment", "With parents",
    ]
    OCCUPATION_TYPE: str
    FLAG_MOBIL: Literal[0, 1] = 1
    FLAG_WORK_PHONE: Literal[0, 1] = 0
    FLAG_PHONE: Literal[0, 1] = 0
    FLAG_EMAIL: Literal[0, 1] = 0
    CNT_FAM_MEMBERS: int = Field(ge=1, le=20)
    EDAD_ANIOS: float = Field(ge=18, le=100)
    DESEMPLEADO: Literal[0, 1] = 0
    ANIOS_EMPLEO: float = Field(ge=0, le=60)


class ResultadoPrediccion(BaseModel):
    probabilidad_mal_pagador: float
    es_riesgo: bool
    umbral_usado: float
    nota: str
