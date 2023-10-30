import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_4.tsv", "r") as file:
    reader = csv.reader(file, delimiter="\t")
    header = next(reader)
    
    for row in reader:
        value = float(row[0])
        values.append(value)

plt.hist(values, bins=6, edgecolor='black', density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram")

plt.show()