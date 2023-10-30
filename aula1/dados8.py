import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_8.dat", "r") as file:
    data = file.read()

# data = [float(value) for value in data]

plt.hist(data, bins=200, edgecolor='black', density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram")

plt.show()
