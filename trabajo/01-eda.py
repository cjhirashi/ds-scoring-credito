import pandas as pd

RUTA = "../datos"

app = pd.read_csv(f"{RUTA}/application_record.csv")
cred = pd.read_csv(f"{RUTA}/credit_record.csv")

print("=== application_record ===")
print("shape:", app.shape)
print("duplicados por ID:", app["ID"].duplicated().sum())
print("nulos:")
print(app.isna().sum()[app.isna().sum() > 0])

print("\n=== credit_record ===")
print("shape:", cred.shape)
print("clientes únicos:", cred["ID"].nunique())
print("rango STATUS_MONTHS_BALANCE:", cred["MONTHS_BALANCE"].min(), cred["MONTHS_BALANCE"].max())
print("nulos:")
print(cred.isna().sum()[cred.isna().sum() > 0])

print("\nvalores de STATUS:")
print(cred["STATUS"].value_counts())

# Definición de "mal pagador" (riesgo): STATUS en {2,3,4,5} = atraso >= 60 días
cred["es_malo"] = cred["STATUS"].isin(["2", "3", "4", "5"]).astype(int)
etiqueta = cred.groupby("ID")["es_malo"].max().reset_index()
print("\nclientes con credit_record:", etiqueta.shape[0])
print("distribución de etiqueta (1 = mal pagador alguna vez):")
print(etiqueta["es_malo"].value_counts())
print("proporción de malos: {:.4f}".format(etiqueta["es_malo"].mean()))

clientes_con_ambos = set(app["ID"]) & set(etiqueta["ID"])
print("\nclientes en application_record:", app["ID"].nunique())
print("clientes con cruce app+credit:", len(clientes_con_ambos))

etiqueta.to_csv("../resultados/etiqueta_riesgo.csv", index=False)
