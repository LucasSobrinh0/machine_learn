import numpy as np
from sklearn.linear_model import LinearRegression

# Dados que representam a quantidade percorrida em KM do carro. 
# A variável x_km será usada como a variável independente (ou feature).
x_km = np.array([20000, 35000, 50000, 65000, 80000, 95000, 110000, 125000, 140000, 155000]).reshape(-1, 1)
# 'reshape(-1, 1)' é necessário porque o método fit() do scikit-learn espera uma entrada em formato de matriz 2D (coluna de dados).
# A forma (-1, 1) significa que o número de linhas é automaticamente ajustado de acordo com o número de elementos, 
# enquanto 1 é o número de colunas. Ou seja, transformamos um vetor 1D em uma matriz 2D de uma única coluna.

# Dados que representam o preço do carro de acordo com o KM percorrido. 
# A variável y_price será a variável dependente (target), ou seja, o preço do carro.
y_price = np.array([85000, 80000, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000])

# Criação do modelo de regressão linear
model_linear_regression = LinearRegression()
# Aqui estamos criando uma instância da classe LinearRegression que representa nosso modelo de regressão linear.

# Ajuste (treinamento) do modelo aos dados
model_linear_regression.fit(x_km, y_price)
# A função fit() ajusta o modelo linear aos dados fornecidos. O modelo aprende a relação entre a variável independente (x_km) 
# e a variável dependente (y_price), ou seja, ele encontra a melhor linha reta que descreve a relação entre a quantidade de KM
# percorridos e o preço do carro. O método fit() utiliza um algoritmo de minimização de erro, como o método dos mínimos quadrados.

# Previsão do preço do carro baseado em uma quantidade de KM fornecida
value_by_km = np.array([30000]).reshape(1, -1)
# Aqui criamos um array para prever o preço de um carro que percorreu 30.000 KM.
# A função reshape(1, -1) é usada para garantir que o valor seja tratado como uma matriz 2D, que é o formato esperado pelo método predict().
# Isso transforma o array de 1D (com apenas um valor) em uma matriz com uma linha e uma coluna, que é o formato correto.

# Previsão do modelo para o valor de 30.000 KM
prediction_value = model_linear_regression.predict(value_by_km)
# O método predict() usa o modelo treinado para prever o valor de y (preço do carro) para os dados de entrada x (KM percorridos).
# Neste caso, o modelo foi treinado com base nos dados de KM e preço, e agora estamos pedindo para prever o preço de um carro 
# que percorreu 30.000 KM.

# Exibe o valor previsto
print(prediction_value)
# Aqui, o resultado da previsão será impresso, mostrando o preço estimado do carro com base no KM fornecido.
