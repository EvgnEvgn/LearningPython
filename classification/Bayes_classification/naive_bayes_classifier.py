import numpy as np
import matplotlib.pyplot as plt


def plot_samples(sample1, sample2):
    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(sample1[0], sample1[1], color='blue', marker='o', s=6)
    plt.scatter(sample2[0], sample2[1], color='red', marker='o', s=6)
    plt.show()


def get_prior_probability(l_y, l):
    return l_y/l


# Исходные данные
mu_1 = [3.0, 5.0]
std_1 = np.matrix([[3.0, 0.0], [0.0, 3.0]], dtype=float)
n = 300
mu_2 = [10.0, 5.0]
std_2 = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float)

samples_1 = np.random.multivariate_normal(mu_1, std_1, n)
samples_2 = np.random.multivariate_normal(mu_2, std_2, n)

x1 = samples_1[:, 0]
y1 = samples_1[:, 1]

x2 = samples_2[:, 0]
y2 = samples_2[:, 1]
plot_samples([x1, y1], [x2, y2])

print(x2)







