"""Gera um dataset fictício: horas de estudo -> nota na prova."""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 200
horas = np.random.uniform(0, 10, n)                # 0 a 10 horas de estudo
ruido = np.random.normal(0, 5, n)                  # variação aleatória
nota = 30 + 6.5 * horas + ruido                    # relação linear + ruído
nota = np.clip(nota, 0, 100)                       # nota entre 0 e 100

df = pd.DataFrame({"horas_estudo": horas.round(2), "nota": nota.round(1)})
df.to_csv("data/estudo_notas.csv", index=False)
print("Arquivo data/estudo_notas.csv criado com", len(df), "linhas.")
