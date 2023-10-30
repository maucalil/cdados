import csv
import matplotlib.pyplot as plt

header = []
values = []
with open("files/dados_5.csv", "r") as file:
    reader = csv.reader(file, delimiter=";")
    header = next(reader)
    
    for row in reader:
        print(row)
        value = float(row[1].replace(",", "."))
        values.append(value)


plt.hist(values, bins=50, edgecolor='black', density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram")

plt.show()
