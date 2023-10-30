import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
import numpy as np

def main():
    A = get_data("dados_A.txt")
    B = get_data("dados_B.txt")

    # getting parameters from data
    muA = np.mean(A)
    sigmaA = np.std(A)
    muB = np.mean(B)
    sigmaB = np.std(B)

    N = 100
    x = np.linspace(min(A), max(A), N)
    y = np.linspace(min(B), max(B), N)

    X = [gaussian(val, muA, sigmaA) for val in x] # valores de A
    Y = [gaussian(val, muB, sigmaB) for val in y] # valores de B

    Z = X * Y
    integral = np.sum(Z[Y > X])
    # for i in range(N):
    #     for j in range(N):
    #       Z.append(X[i] * Y[j])
    #       if Y[j] > X[i]: # P(B > A)
    #           integral += Z[i][j]
        # Create a contour plot
    plt.contour(X, Y, Z, levels=10, colors='black')
    plt.xlabel('X (Gaussian A)')
    plt.ylabel('Y (Gaussian B)')
    plt.title('Contour Plot of Z (Product of Gaussians)')
    plt.colorbar()
    plt.show()
    

def get_data(file):
    PATH = "files/" + file
    data = []
    with open(PATH, "r") as file:
        reader = csv.reader(file, delimiter=" ")
        next(reader)
        
        for row in reader:
            value = float(row[1])
            data.append(value)
    return data
    
def gaussian(x, mu, sigma):
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)

if __name__ == "__main__":
    main()