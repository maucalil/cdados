import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    A = get_data("dados_A.txt")
    B = get_data("dados_B.txt")

    # getting parameters from data
    muA = np.mean(A)
    sigmaA = np.std(A)
    muB = np.mean(B)
    sigmaB = np.std(B)

    min_value = min(min(A), min(B))
    max_value = max(max(A), max(B))

    x = np.linspace(min_value, max_value, 1000)
    
    yA = [gaussian(val, muA, sigmaA) for val in x]
    yB = [gaussian(val, muB, sigmaB) for val in x]
    
    plt.plot(x, yA, label='A', color="r")
    plt.plot(x, yB, label='B', color="b")
    
    plt.xlabel('X')
    plt.ylabel('Gaussian Value')
    plt.legend()
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