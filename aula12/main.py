import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
num_points = 100

x_a = np.random.uniform(1, 50, num_points)
x_b = np.random.uniform(50, 100, num_points)
y_b = np.random.uniform(1, 50, num_points)
y_a = np.random.uniform(50, 100, num_points)

dataA = np.array([[x, y] for x,y in zip(x_a, y_a)])
dataB = np.array([[x, y] for x, y in zip(x_b, y_b) ])

w = np.array([1.0, 5.0])

def print_graphics(dataA, dataB):
  plt.scatter(dataA[:, 0], dataA[:, 1], label='Class A', color='blue')
  plt.scatter(dataB[:, 0], dataB[:, 1], label='Class B', color='red')

  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.legend()

  plt.show()