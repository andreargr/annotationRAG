import pandas as pd

# Cargar los CSV
df1 = pd.read_csv('../10K/df_gpt_5_mini_inference_index_k10_filtrado.csv', header=0)
df2 = pd.read_csv('../5K/all_df_5_mini_inference_index.csv', header=0)

# Filtrar df2 para que solo contenga Labels que estén en df1
df2_filtered = df2[df2['Label'].isin(df1['Label'])]

# Guardar el resultado
df2_filtered.to_csv('100_df_5_mini_inference_index.csv', index=False)

print(f"Filas originales df2: {len(df2)}, filas filtradas: {len(df2_filtered)}")
