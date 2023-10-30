import numpy as np
import matplotlib.pyplot as plt

def main():
  data = np.random.exponential(scale=2.5, size=100)

  plt.hist(data, edgecolor='black')
  plt.xlabel("Values")
  plt.ylabel("Frequency")
  plt.title("Histogram")
  plt.show()

  print(1/np.mean(data))

if __name__ == "__main__":
  main()