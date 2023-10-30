import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def main():
  data_frac = get_data("dados_frac_V.txt")
  data_porc = get_data("dados_porcento_U.txt")

  data_frac /= 100
  data_porc /= 100

  x = np.linspace(0, 1, num=50)

  beta_pdf = beta.pdf(x, 2, 5)

  N = 50
  alphas = np.linspace(0, 5, N)
  betas = np.linspace(0, 5, N)

  bestL = float("-inf")
  bestAlpha = 0
  bestBeta = 0
  for i in range(N):
    for j in range(N):
      L = 0
      for k in range(len(data_frac)):
        L += beta.logpdf(data_frac[k], alphas[i], betas[j]) # soma por ser log

      if L > bestL:
        bestL = L
        bestAlpha = alphas[i]
        bestBeta = betas[j]

  print(f"Best alpha = {bestAlpha}")
  print(f"Best beta = {bestBeta}")


def plot_histogram(data, bins=50, label='', color='blue'):
    plt.hist(data, bins, label=label, color=color, alpha=0.5)
    plt.legend()
    plt.show()

def get_data(file):
  PATH = "files/" + file
  return np.loadtxt(PATH, delimiter=" ", skiprows=1, usecols=(1,))

if __name__ == "__main__":
    main()
