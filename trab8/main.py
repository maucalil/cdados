import pandas as pd
import matplotlib.pyplot as plt
from pvclust import PvClust

data = pd.read_csv("GSE179175_filtered.csv", delimiter="\t", header=0)

X = data.iloc[:10000, 1:]

# Por algum motivo o r é divido por 10, por isso foi passado [8, 10, 12]
pv = PvClust(X, method='average', metric='euclidean', r=[8, 10, 12], nboot=1000, parallel=False)
pv.plot(labels=X.columns)
plt.show()