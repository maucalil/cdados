import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_3.tsv", "r") as file:
    reader = csv.reader(file, delimiter="\t")
    header = next(reader)
    
    for row in reader:
        print(row)
        value = float(row[1])
        values.append(value)


plt.hist(values, edgecolor='black', density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram")

plt.show()
