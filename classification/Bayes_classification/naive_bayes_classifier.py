import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from itertools import product
import math


def plot_samples(sample1, sample2, sample3):

    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(sample1[:, 0], sample1[:, 1], color='blue', marker='o', s=6)
    plt.scatter(sample2[:, 0], sample2[:, 1], color='red', marker='o', s=6)
    if len(sample3) > 0:
        plt.scatter(sample3[:, 0], sample3[:, 1], color='black', marker='o', s=6)
    plt.show()


def get_points(x_left, x_right, y_left, y_right, step):
    return list(product(np.arange(x_left, x_right, step), np.arange(y_left, y_right, step)))


def get_prior_probability(l_y, l):

    return l_y/l


def get_rectangular_kernel(z):

    return 0.5 * (abs(z) <= 1)


def get_kernel(x, y, h):
    z = (x - y) / h

    return get_rectangular_kernel(z)


def get_columns(array):
    result = []
    row, col = np.shape(array)
    current = 0
    while current < col:
        result.append(array[:, current])
        current += 1

    return result


def get_parzen_rozenblatt_probability_estimate(value_of_x_attribute, values_of_x_attribute, h):
    m = len(values_of_x_attribute)
    #kernels = list(map(get_kernel, list(repeat(value_of_x_attribute, m)), values_of_x_attribute, list(repeat(h, m))))
    kernels = [get_kernel(value_of_x_attribute, y_attr, h) for y_attr in values_of_x_attribute]
    kernels_sum = np.sum(kernels)
    probability_estimate = kernels_sum / (m * h)

    return probability_estimate


def gamma(x_object, array_of_x_objects, array_h, lambda_y, l_y, l):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    array_of_x_objects = get_columns(array_of_x_objects)

    #probability_estimates = list(map(get_parzen_rozenblatt_probability_estimate, x_object, array_of_x_objects, array_h))
    probability_estimates = []
    index = 0
    for attribute in x_object:
        estimate = get_parzen_rozenblatt_probability_estimate(attribute, array_of_x_objects[index], array_h[index])
        index += 1
        probability_estimates.append(estimate)

    if any(x == 0.0 for x in probability_estimates):
        return 0
    c = np.sum(list(map(math.log, probability_estimates)))

    return b + c


def classify(all_objects, classes):
    lambda_y = 1
    length_of_all_objects = len(all_objects)
    h_default = 1
    result_classes = {"666": []}
    a_values = {}
    h = {}
    not_determined_class = "666"

    for cls, objects in classes.items():
        result_classes.setdefault(cls, [])
        a_values.setdefault(cls, 0)
        row, col = np.shape(objects)
        h_ = list(repeat(h_default, col))
        #h_ = [h_default * int(cls) + 0.05, h_default * int(cls) + 0.1]
        h.setdefault(cls, h_)

    for x in all_objects:
        for cl, class_objects in classes.items():
            l_y = len(class_objects)
            g = gamma(x, class_objects, h.get(cl), lambda_y, l_y, length_of_all_objects)
            a = {cl: g}
            a_values.update(a)
        if a_values["1"] == 0 and a_values["2"] == 0:
            best_class = not_determined_class
        elif a_values["1"] == 0:
            best_class = "2"
        elif a_values["2"] == 0:
            best_class = "1"
        else:
            best_class = max(a_values, key=a_values.get)
        objects_in_classes = result_classes.get(best_class)
        objects_in_classes.append(x)
        result_classes.update({best_class: objects_in_classes})

    return result_classes


# Исходные данные
mu_1 = [3.0, 5.0]
std_1 = np.matrix([[3.0, 0.0], [0.0, 3.0]], dtype=float)
n = 100
mu_2 = [10.0, 5.0]
std_2 = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float)

samples_1 = np.random.multivariate_normal(mu_1, std_1, n)
samples_2 = np.random.multivariate_normal(mu_2, std_2, n)

x1_y1 = samples_1[:, 0]
x2_y1 = samples_1[:, 1]

x1_y2 = samples_2[:, 0]
x2_y2 = samples_2[:, 1]
init_classes = {"1": samples_1, "2": samples_2}
#plot_samples(samples_1, samples_2)

#all_points = np.concatenate((samples_1, samples_2), axis=0)
all_points = get_points(-3, 14, 0, 13, 0.4)
classificated_points = classify(all_points, init_classes)
not_determined = np.array(classificated_points["666"], dtype=float)

plot_samples(np.array(classificated_points["2"]), np.array(classificated_points["1"]), [])

