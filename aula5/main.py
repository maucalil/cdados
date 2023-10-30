import numpy as np
import matplotlib.pyplot as plt

# order (mu, sigma, n)
A = np.random.normal(10, 3, 80) # só deus sabe que mu=10 (media) sigma=3 (std)
B = np.random.normal(15, 3.5, 60) # só deus sabe que mu=15 (media) sigma=3.5 (std)

# getting parameters from data
muA = np.mean(A)
sigmaA = np.std(A)
muB = np.mean(B)
sigmaB = np.std(B)

# agora que já aprendi, posso sar o modelo como quiser
AFake = np.random.normal(muA, sigmaA, 100000)
BFake = np.random.normal(muB, sigmaB, 100000)

# plt.hist(AFake, bins=50, color='pink', label="AFake")
# plt.hist(BFake, bins=50, color='black', label="BFake")
# plt.show()

delta = AFake - BFake
R = AFake / BFake

# Calculate the probability that delta is greater than 0
probability_delta_gt_0 = np.sum(delta > 0) / len(delta)
print("P(Delta > 0):", probability_delta_gt_0)

# Calculate the probability that delta is greater than 0
probability_R_gt_1 = np.sum(R > 1) / len(R)
print("P(R > 1):", probability_R_gt_1)

plt.hist(delta, bins=120)
plt.show()