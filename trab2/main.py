import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Be sure the file path is correct
    A = get_data("files/dados_A.txt")
    B = get_data("files/dados_B.txt")

    # getting parameters from data
    muA = np.mean(A)
    sigmaA = np.std(A)
    muB = np.mean(B)
    sigmaB = np.std(B)

    N = 1000
    # separating A and B data in linear spaces
    x = np.linspace(min(A), max(A), N)
    y = np.linspace(min(B), max(B), N)

    # applying the function to all values in x and y
    X = [gaussian(val, muA, sigmaA) for val in x]
    Y = [gaussian(val, muB, sigmaB) for val in y]

    Z = np.zeros((N, N)) # start Z as a NxN matrix filled with 0s

    integral = 0
    for i in range(N):
        for j in range(N):
            Z[i, j] = X[i] * Y[j]
            if X[i] > Y[j]: # A > B
                integral += Z[i, j]

    prob_a_greater_b = integral / np.sum(Z) # normalize the probability
    print(f"P[A > B] = {prob_a_greater_b}")
    
    plt.contour(x, y, Z)
    plt.xlabel('X (Gaussian A)')
    plt.ylabel('Y (Gaussian B)')
    plt.title('Contour Plot of Z (Product of Gaussians)')
    plt.colorbar()
    plt.show()
    

def get_data(path):
    data = []
    with open(path, "r") as file:
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