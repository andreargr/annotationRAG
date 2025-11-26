import pandas as pd

# Cargar el archivo (ajusta el nombre según tu caso)
df = pd.read_csv("../candidate_positions.tsv", sep="\t")

# Ver las primeras filas
print(df.head())

conteo_clo = df.groupby(["Type", "UBERON"]).size().reset_index(name="count")
print(conteo_clo)
