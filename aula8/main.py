import numpy as np
from scipy.stats import multivariate_normal

def get_mvtnorm_pdf(x, mi, sigma, k=2):
  x = np.array(x)
  mi = np.array(mi)
  sigma = np.array(sigma)

  const = (2 * np.pi) ** (-k/2)

  det_sigma = np.linalg.det(sigma)
  x_minus_mi = x - mi
  x_minus_mi_t = np.transpose(x_minus_mi)
  inv_sigma = np.linalg.inv(sigma)

  mult = np.matmul(x_minus_mi_t, inv_sigma)
  mult = np.matmul(mult, x_minus_mi)
  exp = np.exp(-0.5 * mult)

  pdf = const * (1 / np.sqrt(det_sigma)) * exp

  return pdf

x = [0.5, 0.5]
mi = [42, 17]
sigma = [[73, -50], [-50, 45]]

my_pdf = get_mvtnorm_pdf(x, mi, sigma)
net_pdf = multivariate_normal.pdf(x, mi, sigma)

print(f"My mvtnorm pdf: {my_pdf:.6f}")
print(f"Internet mvtnorm pdf: {net_pdf:.6f}")

  