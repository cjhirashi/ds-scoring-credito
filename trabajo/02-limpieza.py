import numpy as np
import pandas as pd

RUTA_DATOS = "../datos"
RUTA_RES = "../resultados"

app = pd.read_csv(f"{RUTA_DATOS}/application_record.csv")
etiqueta = pd.read_csv(f"{RUTA_RES}/etiqueta_riesgo.csv")

# Quitar duplicados de ID en application_record (quedarse con la primera ocurrencia)
app = app.drop_duplicates(subset="ID", keep="first")

# Solo clientes con historial en credit_record (los que sí se pueden etiquetar)
df = app.merge(etiqueta, on="ID", how="inner")
print("clientes tras cruce:", df.shape[0])

# OCCUPATION_TYPE: nulo != dato faltante real, suele significar "sin ocupación declarada"
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Sin_declarar")

# Edad y antigüedad laboral en años (DAYS_BIRTH y DAYS_EMPLOYED vienen negativos)
df["EDAD_ANIOS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
# DAYS_EMPLOYED tiene centinela 365243 para desempleados/pensionados
df["DESEMPLEADO"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
df["ANIOS_EMPLEO"] = np.where(df["DAYS_EMPLOYED"] > 0, 0, (-df["DAYS_EMPLOYED"] / 365.25).round(1))

df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])

print("\nnulos restantes:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nEDAD_ANIOS describe:")
print(df["EDAD_ANIOS"].describe().round(1))

print("\nANIOS_EMPLEO describe (excluyendo desempleados):")
print(df.loc[df["DESEMPLEADO"] == 0, "ANIOS_EMPLEO"].describe().round(1))

print("\nshape final:", df.shape)
df.to_csv(f"{RUTA_RES}/dataset_limpio.csv", index=False)
