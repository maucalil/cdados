import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_2.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        value = float(row[1])
        values.append(value)


plt.hist(values, edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram")

plt.show()
