#Link Dados
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5411053

#pegando dados (fora do R)
# cat GSE179175.csv | sed "s/\.[0-9][0-9]/&__/g" | sed "s/__[0-9]*e-/e-/g" | sed "s/__[0-9]*\t/\t/g" > GSE179175_small.csv

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Leitura do arquivo CSV
x = pd.read_csv("GSE179175_small.csv", delimiter="\t", header=0)

# Remoção da primeira coluna
x = x.iloc[:, 1:]

# Seleção das colunas que não tem dados faltantes
x = x.iloc[:, list(range(0, 114, 2))]

# Converte todos os valores para o tipo float
x = x.astype('float64')

# Remoção dos infinitos em cada coluna
for col in x.columns:
    # Encontrar o maior valor não infinito
    max_val = x.loc[x[col] != np.inf, col].max()
    
    # Encontrar o menor valor não infinito
    min_val = x.loc[x[col] != -np.inf, col].min()
    
    x[col].replace({np.inf: max_val + 0.01, -np.inf: min_val - 0.01}, inplace=True)

# Medir distâncias euclidianas entre as colunas (pacientes)
D = pdist(x.T)
D = squareform(D) # converte o vetor de distancias para matriz

# Visualizar a matriz de distâncias
plt.imshow(D) # origin='lower' se quiser ver igual no R
plt.colorbar(label='Distância Euclidiana')
plt.show()

# Árvore de cluster (hierarchical clustering)
h = linkage(x.T, method='complete') # complete é o padrão no R

# Visualizar o dendrograma
dendrogram(h, labels=x.columns, distance_sort='descending') # os de maior distancia estão mais a esquerda
plt.title('Dendrograma Hierárquico')
plt.xlabel('Pacientes')
plt.ylabel('Distância Euclidiana')
plt.show()


