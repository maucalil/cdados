from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

import random

def gerar_sequencia_aleatoria(tamanho):
    sequencia_aleatoria = [random.uniform(500, 1000) for _ in range(tamanho)]
    return sequencia_aleatoria

lenght = 1000
seq = gerar_sequencia_aleatoria(lenght)

# Transformar em distribuição normal usando numpy
seq_norm = norm.pdf(seq, 750, 100)

# Plotar um único gráfico
plt.figure(figsize=(10, 6))
plt.scatter(seq, seq_norm, alpha=0.5)
plt.xlabel('Sequência de Floats')
plt.ylabel('Sequência Normalizada')
plt.grid(True)
plt.show()