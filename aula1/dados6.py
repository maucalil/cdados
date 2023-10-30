import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_6.txt", "r") as file:
    data = file.read().strip().split(",")
    data.pop()
data = [float(value) for value in data]

plt.hist(data, bins=100, edgecolor='black', density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram")

plt.show()
