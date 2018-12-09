import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import math

def plot_samples(sample1, sample2):
    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(sample1[0], sample1[1], color='blue', marker='o', s=6)
    plt.scatter(sample2[0], sample2[1], color='red', marker='o', s=6)
    plt.show()


def get_prior_probability(l_y, l):

    return l_y/l


def get_rectangular_kernel(z):

    return 0.5 * (abs(z) <= 1)


def get_kernel(x, y, h):
    z = (x - y) / h

    return get_rectangular_kernel(z)


def get_parzen_rozenblatt_probability_estimate(value_of_x_attribute, values_of_x_attribute, h):
    m = len(values_of_x_attribute)
    kernels = list(map(get_kernel, repeat(value_of_x_attribute, m), values_of_x_attribute, repeat(h, m)))
    kernels_sum = np.sum(kernels)
    probability_estimate = kernels_sum / (m * h)

    return probability_estimate


def gamma(x_object, array_of_x_objects, array_h, lambda_y, l_y, l):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    probability_estimates = list(map(get_parzen_rozenblatt_probability_estimate, x_object, array_of_x_objects, array_h))
    c = np.sum(map(math.log, probability_estimates))

    return b + c

def classify(classes, samples_of_classes):
    lambda_y = 1

    class_amount = len(classes)
    result = []
    for cl in classes:
        class_points =

        h = repeat()




# Исходные данные
mu_1 = [3.0, 5.0]
std_1 = np.matrix([[3.0, 0.0], [0.0, 3.0]], dtype=float)
n = 300
mu_2 = [10.0, 5.0]
std_2 = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float)

samples_1 = np.random.multivariate_normal(mu_1, std_1, n)
samples_2 = np.random.multivariate_normal(mu_2, std_2, n)

x1_y1 = samples_1[:, 0]
x2_y1 = samples_1[:, 1]

x1_y2 = samples_2[:, 0]
x2_y2 = samples_2[:, 1]
#plot_samples([x1, y1], [x2, y2])


