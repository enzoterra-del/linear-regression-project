"""Regressão Linear simples: prever a nota de uma prova pelas horas de estudo."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Carregar os dados com pandas
df = pd.read_csv("data/estudo_notas.csv")
print("Primeiras linhas:")
print(df.head(), "\n")
print("Resumo estatístico:")
print(df.describe(), "\n")

# 2. Separar X (entrada) e y (o que queremos prever)
X = df[["horas_estudo"]]   # duplo colchete = DataFrame (o sklearn exige 2D)
y = df["nota"]

# 3. Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

# 4. Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print(f"Equação aprendida: nota = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f} * horas_estudo\n")

# 5. Avaliar no conjunto de teste
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Métricas (teste) ===")
print(f"R² (\"acurácia\" da regressão): {r2:.2%}")
print(f"MAE  (erro médio absoluto):    {mae:.2f} pontos")
print(f"RMSE (raiz do erro quadrático): {rmse:.2f} pontos\n")

# 6. Gráficos com matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 6.1 Dados de treino/teste + reta do modelo
axes[0].scatter(X_train, y_train, alpha=0.6, label="Treino")
axes[0].scatter(X_test, y_test, alpha=0.8, color="orange", label="Teste")
linha_x = pd.DataFrame({"horas_estudo": np.linspace(0, 10, 100)})
axes[0].plot(linha_x, modelo.predict(linha_x), color="red", label="Reta do modelo")
axes[0].set_xlabel("Horas de estudo")
axes[0].set_ylabel("Nota")
axes[0].set_title("Regressão Linear")
axes[0].legend()

# 6.2 Real vs previsto
axes[1].scatter(y_test, y_pred, alpha=0.8)
lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[1].plot(lim, lim, "r--", label="Previsão perfeita")
axes[1].set_xlabel("Nota real")
axes[1].set_ylabel("Nota prevista")
axes[1].set_title(f"Real vs Previsto (R² = {r2:.2f})")
axes[1].legend()

plt.tight_layout()
plt.savefig("resultado.png", dpi=150)
plt.show()

# 7. Fazer uma previsão nova
horas_novas = pd.DataFrame({"horas_estudo": [7.5]})
print(f"Previsão para 7,5 horas de estudo: {modelo.predict(horas_novas)[0]:.1f}")
