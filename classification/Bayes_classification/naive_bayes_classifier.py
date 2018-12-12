import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from itertools import product
import math
import time


def plot_samples(sample1, sample2, sample3=None, classes=None, boundary_points=None):

    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(sample1[:, 0], sample1[:, 1], c='blue', marker="+", s=70)
    plt.scatter(sample2[:, 0], sample2[:, 1], c='red', marker="+", s=70)

    if len(sample3) > 0:
        plt.scatter(sample3[:, 0], sample3[:, 1], c='grey', marker="+", s=70)

    if len(classes) > 0:
        colors = ['red', 'blue', 'green', 'orange', 'yellow', 'black']
        i = 0
        for cl, points in classes.items():
            plt.scatter(points[:, 0], points[:, 1], c=colors[i], marker="o", s=20)
            i += 1

    if len(boundary_points) > 0:
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='black')

    plt.show()


def get_boundary(b_points):
    b_points.sort(axis=0)
    return b_points


def get_points(x_left, x_right, y_left, y_right, step):
    return list(product(np.arange(x_left, x_right, step), np.arange(y_left, y_right, step)))


def column(matrix, i):
    return [row[i] for row in matrix]


def get_columns(matrix):
    row, col = np.shape(matrix)
    return [column(i) for i in range(col)]


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
    kernels = [get_kernel(value_of_x_attribute, y_attr, h) for y_attr in values_of_x_attribute]
    #kernels = list(map(get_kernel, repeat(value_of_x_attribute, m), values_of_x_attribute, repeat(h, m)))
    kernels_sum = sum(kernels)
    probability_estimate = kernels_sum / (m * h)

    return probability_estimate


def gamma(x_object, array_of_x_objects, array_h, lambda_y, l_y, l):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    x_objects = get_columns(array_of_x_objects)

    probability_estimates = []
    index = 0
    for attribute in x_object:
        estimate = get_parzen_rozenblatt_probability_estimate(attribute, x_objects[index], array_h[index])
        index += 1
        probability_estimates.append(estimate)
    #probability_estimates = list(map(get_parzen_rozenblatt_probability_estimate, x_object, x_objects, array_h))

    if any(x == 0.0 for x in probability_estimates):
        return 0.0
    c = sum([math.log(pe) for pe in probability_estimates])
    #c = np.sum(list(map(math.log, probability_estimates)))

    return b + c


def classify(all_objects, classes):
    start_time = time.process_time()
    lambda_y = 1
    length_of_all_objects = len(all_objects)
    h_default = 1
    result_classes = {"not_determined": []}
    a_values = {}
    h = {}
    boundary_points = []
    gammas = {}
    not_determined_class = "not_determined"
    # точность для разности гамм
    epsilon = 1

    for cls, objects in classes.items():
        result_classes.setdefault(cls, [])
        a_values.setdefault(cls, 0.0)
        row, col = np.shape(objects)
        h_ = list(repeat(h_default, col))
        h.setdefault(cls, h_)
        gammas.setdefault(cls, 0.0)

    for x in all_objects:

        for cl, class_objects in classes.items():
            l_y = len(class_objects)
            g = gamma(x, class_objects, h.get(cl), lambda_y, l_y, length_of_all_objects)
            a = {cl: g}
            a_values.update(a)
            gammas.update({cl: abs(g)})
        # убираем нулевые значения
        a_values = {k: v for k, v in a_values.items() if v != 0.0}

        if len(a_values) == 0:
            best_class = not_determined_class
        else:
            # допущение, что класса всего два (хардкод)
            difference = abs(gammas.get("1") - gammas.get("2"))
            if difference <= epsilon:
                boundary_points.append(x)
            best_class = max(a_values, key=a_values.get)
        objects_in_classes = result_classes.get(best_class)
        objects_in_classes.append(x)
        result_classes.update({best_class: objects_in_classes})



    end_time = time.process_time()
    print(end_time - start_time)

    return result_classes, boundary_points


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
init_classes = {"1": samples_1, "2": samples_2}
#plot_samples(samples_1, samples_2)

#all_points = np.concatenate((samples_1, samples_2), axis=0)
all_points = get_points(-10, 20, -10, 20, 0.2)
# times = 10
# while times > 0:
classified_points, bound_points = classify(all_points, init_classes)

bound_points = np.array(bound_points)
bound_points = get_boundary(bound_points)

not_determined = np.array(classified_points["not_determined"], dtype=float)
plot_samples(np.array(classified_points["2"]), np.array(classified_points["1"]), not_determined, init_classes, bound_points)

