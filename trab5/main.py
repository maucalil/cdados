import numpy as np
from scipy.stats import multivariate_normal

def main():
    k = 3
    x = np.array([0.5, 0.5, 0.5])
    mi = np.array([0, 100, 50]) # media
    sigma = np.array([[64.9, 33.2, -24.4],
                      [33.2, 56.4, -24.1],
                      [-24.4, -24.1, 75.6]]) # matriz de covariancia

    # Comparando a minha implementacao da funcao normal multivariada com a do pacote
    my_pdf = get_mvtnorm_pdf(x, mi, sigma, k)
    net_pdf = multivariate_normal.pdf(x, mi, sigma)

    print(f"My mvtnorm pdf: {my_pdf:.6f}")
    print(f"Internet mvtnorm pdf: {net_pdf:.6f}")
    print("-")

    # Simulando dados que seguem uma mvtnorm 3-d com parâmetros da minha cabeca
    N = 10000
    data = np.load(f"aula9/dados_{N}.npy")
    # data = np.random.multivariate_normal(mean=mi, cov=sigma, size=N)
    # np.save(f"aula9/dados_{N}", data)
    # np.savetxt(f"aula9/dados_{N}.txt", data, fmt="%.6f")

    mi_est = estimate_mi(data)
    sigma_est = estimate_sigma(data)
    print(f"MI estimated:\n {mi_est}")
    print(f"Sigma estimated (cov matrix):\n {sigma_est}")
    print(f"Correlation matrix:\n {np.corrcoef(data, rowvar=False)}")

def get_mvtnorm_pdf(x, mi, sigma, k):
  const = (2 * np.pi) ** (-k/2)

  det_sigma = np.linalg.det(sigma)
  x_minus_mi = x - mi
  inv_sigma = np.linalg.inv(sigma)

  mult = np.matmul(x_minus_mi.T, inv_sigma)
  mult = np.matmul(mult, x_minus_mi)
  exp = np.exp(-0.5 * mult)

  pdf = const * (1 / np.sqrt(det_sigma)) * exp

  return pdf

def estimate_mi(X):
   return np.mean(X, axis=0)

def estimate_sigma(X):
    return np.cov(X, rowvar=False)

if __name__ == "__main__":
   np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
   main()