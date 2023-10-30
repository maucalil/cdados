import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def main():
    mi_A = np.array([0, 100]) # media
    sigma_A = np.array([[64.9, 33.2],
                        [33.2, 56.4]]) # matriz de covariancia
    
    mi_B = np.array([-10, 33]) # media
    sigma_B = np.array([[38.5, -24.2],
                        [-24.2, 73.1]]) # matriz de covariancia
    
    # Crie um grid de pontos de teste em 2D
    x_range = np.linspace(-200, 100, 100)  # Substitua os valores e a resolução conforme necessário
    y_range = np.linspace(-75, 200, 100)
    grid = np.array([(x, y) for x in x_range for y in y_range])

    classifications = []

    for point in grid:
        prob_A = multivariate_normal.pdf(point, mean=mi_A, cov=sigma_A)
        prob_B = multivariate_normal.pdf(point, mean=mi_B, cov=sigma_B)

        if prob_A > prob_B:
            classifications.append("A")
        else:
            classifications.append("B")

if __name__ == "__main__":
   np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
   main()