import numpy as np
import matplotlib.pyplot as plt

def main():
  np.random.seed(42)

  # Definindo os coeficientes de um polinomio de terceiro grau
  coef = np.random.rand(4)
  print("Coeficientes reais:", coef)

  # Gerando 100 pontos aleatorios e adicionando ruido
  n = 100
  x = np.linspace(-10, 10, n)
  x = x + np.random.normal(0, 0.5, n)

  # Calculando os valores de y usando o polinômio e adicionando ruido
  y = np.polyval(coef, x)
  y = [val + np.random.normal(0, 0.2*abs(val)) 
                  for val in y]

  # Calculando a soma dos residuos
  coef_test = coef - 0.09
  print("Coeficientes teste:", coef_test)
  sum = soma_residuos(coef_test, x, y)

  x_test = np.linspace(-10, 10, 500)
  y_test = np.polyval(coef_test, x_test)
  print(x_test.shape, y_test.shape)

  # Plotando os dados
  plt.scatter(x, y, label='Dados com Ruído', s=5)
  plt.plot(x_test, y_test, label='Dados do Modelo')

  plt.title('Dados')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.show()

def soma_residuos(coeficientes, x_vals, y_vals):
  soma = 0
  for x, y_real in zip(x_vals, y_vals):
    y_pred = np.polyval(coeficientes, x)
    residuo = (y_real - y_pred) ** 2
    soma += residuo

  return soma

main()