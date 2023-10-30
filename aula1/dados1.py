import csv
import matplotlib.pyplot as plt

data = []
with open("files/dados_1.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.extend([float(item) for item in row])

plt.hist(data, edgecolor='black')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram")

plt.show()
