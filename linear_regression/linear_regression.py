# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados fornecidos
ano = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
valor = np.array([4.19, 5.85, 5.75, 5.67, 5.27, 5.73])

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(ano, valor)

# Previsão para os anos futuros
anos_futuros = np.array([2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
previsoes = modelo.predict(anos_futuros)

# Visualizando os dados e a linha de regressão
plt.scatter(ano, valor, color='blue', label='Dados reais')
plt.plot(ano, modelo.predict(ano), color='red', label='Linha de Regressão')
plt.scatter(anos_futuros, previsoes, color='green', label='Previsões')
plt.xlabel('Ano')
plt.ylabel('Valor do Dólar')
plt.title('Previsão do Valor do Dólar com Regressão Linear')
plt.legend()
plt.show()

# Exibindo previsões futuras
for ano, previsao in zip(anos_futuros.flatten(), previsoes):
    print(f'Previsão para {ano}: {previsao:.2f}')
