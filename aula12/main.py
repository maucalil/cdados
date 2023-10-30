import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(42)
    num_points = 100

    mean_A = [25, 75]
    cov_A = [[100, 0], [0, 100]]

    mean_B = [75, 25]
    cov_B = [[100, 0], [0, 100]]

    dataA = np.random.multivariate_normal(mean_A, cov_A, num_points)
    dataB = np.random.multivariate_normal(mean_B, cov_B, num_points)

    best_w = None
    best_error = float("inf")

    for w1 in range(0, 1):
        for w2 in range(-50, 51):
            w = np.array([w1, w2])
            errorA = calc_error(dataA, w)
            errorB = calc_error(dataB, w, positive=False)
            error = (errorA + errorB) / 2

            if error < best_error:
                best_error = error
                best_w = w

    print(f"Melhores coeficientes: w1 = {best_w[0]}, w2 = {best_w[1]}")
    print(f"Erro mínimo: {best_error}")
    print_graphics(dataA, dataB, best_w)

def print_graphics(dataA, dataB, w=None):
    plt.scatter(dataA[:, 0], dataA[:, 1], label="Class A", color="blue")
    plt.scatter(dataB[:, 0], dataB[:, 1], label="Class B", color="red")


    if w is not None:
        x_boundary = np.linspace(0, 100, 100)
        y_boundary = (-w[0] / w[1]) * x_boundary
        plt.plot(x_boundary, y_boundary, label='Boundary', color='green')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    plt.show()

def calc_error(data, w, positive = True):
    if positive:
        classes = np.dot(data, w) > 0 # pontos acima da reta
    else:
        classes = np.dot(data, w) <= 0 # pontos abaixo da reta
    
    errors = np.sum(classes is False)

    return errors/len(classes)


if __name__ == "__main__":
    main()
