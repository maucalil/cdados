import numpy as np
import matplotlib.pyplot as plt

nA = 50
nB = 43

A = np.random.normal(9.5, 1.0, nA)
B = np.random.normal(6.0, 2.0, nB)

# Ok, agora temos um modelo teórico. Adeus dados!
muA = np.mean(A)
sigmaA = np.std(A)

muB = np.mean(B)
sigmaB = np.std(B)

D = muA - muB # ta mas... e a estcasticidade??

S = int(1E6)
# D = np.full(S, np.inf) # diferença entre as medias

# for i in range(S):
#   a = np.random.normal(muA, sigmaA, nA)
#   b = np.random.normal(muB, sigmaB, nB)
#   D[i] = np.mean(a) - np.mean(b)

# # Save the D array to a file
# np.save('D.npy', D)

# Load the D array from the file
D = np.load('D.npy')

hist_values, bin_edges = np.histogram(D, bins=50, density=True)
delta = bin_edges[1] - bin_edges[0]
mid_index = int(len(hist_values)/2)
current_sum = hist_values[mid_index] * delta
target_percentage = 0.95
i = 1

while current_sum < target_percentage:
  current_sum += (hist_values[mid_index - i] * delta) + (hist_values[mid_index + i] * delta)
  i += 1

left_index = mid_index - (i - 1)
right_index = mid_index + (i - 1)
  
print(f"Interval from {bin_edges[left_index]:.6f} to {bin_edges[right_index]:.6f}")
print(f"Soma = {current_sum}")

plt.hist(D, bins=50)
plt.axvline(bin_edges[left_index], color='r', linestyle='dashed', linewidth=2, label='Left Edge')
plt.axvline(bin_edges[right_index], color='g', linestyle='dashed', linewidth=2, label='Right Edge')
plt.title("Histogram of D")
plt.legend()
plt.show()



