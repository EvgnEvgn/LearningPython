import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from itertools import product
import math
import time
from numpy import sin, cos, pi
from scipy.optimize import leastsq
from sys import float_info

def find_boundary(x, y, n, plot_pts=1000):
    def sines(theta):
        ans = np.array([sin(i * theta) for i in range(n + 1)])
        return ans

    def cosines(theta):
        ans = np.array([cos(i * theta) for i in range(n + 1)])
        return ans

    def residual(params, x, y):
        x0 = params[0]
        y0 = params[1]
        c = params[2:]

        r_pts = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5

        thetas = np.arctan2((y - y0), (x - x0))
        m = np.vstack((sines(thetas), cosines(thetas))).T
        r_bound = m.dot(c)

        delta = r_pts - r_bound
        delta[delta > 0] *= 10

        return delta

    # initial guess for x0 and y0
    x0 = x.mean()
    y0 = y.mean()

    params = np.zeros(2 + 2 * (n + 1))
    params[0] = x0
    params[1] = y0
    params[2:] += 1000

    popt = leastsq(residual, x0=params, args=(x, y), ftol=1.e-12, xtol=1.e-12)[0]

    thetas = np.linspace(0, 2 * pi, plot_pts)
    m = np.vstack((sines(thetas), cosines(thetas))).T
    c = np.array(popt[2:])
    r_bound = m.dot(c)
    x_bound = x0 + r_bound * cos(thetas)
    y_bound = y0 + r_bound * sin(thetas)

    return x_bound, y_bound


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
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], c='black', lw=2)

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
    return l_y / l


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
    kernels_sum = sum(kernels)
    probability_estimate = kernels_sum / (m * h)

    return probability_estimate


def gamma(x_object, array_of_x_objects, h, lambda_y, l_y, l):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    x_objects = get_columns(array_of_x_objects)

    probability_estimates = []
    index = 0
    for attribute in x_object:
        estimate = get_parzen_rozenblatt_probability_estimate(attribute, x_objects[index], h)
        index += 1
        probability_estimates.append(estimate)

    if any(x == 0.0 for x in probability_estimates):
        return -float_info.max
    else:
        c = sum([math.log(pe) for pe in probability_estimates])

    return b + c


def leave_one_out(classes):
    start_time = time.process_time()

    h = 0.9
    h_step = 0.1
    h_stop = 6
    lambda_y = 1
    all_objects = []
    a_values_for_classes = {}

    for cls, class_objects in classes.items():
        all_objects.append(class_objects)
        a_values_for_classes.setdefault(cls, 0.0)

    loo_result = {}

    all_objects = np.concatenate(all_objects)
    length_of_all_objects = len(all_objects)

    while h <= h_stop:
        a = []
        for obj in all_objects:
            true_class_for_object = ""
            for cls, class_objects in classes.items():
                obj = list(obj)
                objects = class_objects.copy()

                if obj in objects:
                    true_class_for_object = cls
                    objects.remove(obj)

                l_y = len(objects)
                g = gamma(obj, np.array(objects), h, lambda_y, l_y, length_of_all_objects)
                a_values_for_classes.update({cls: g})

            #a_values_for_classes = {k: v for k, v in a_values_for_classes.items() if v != 0.0}

            #if len(a_values_for_classes) == 0:
            #    a.append(1)
            if len(a_values_for_classes) > 1 and len(set(a_values_for_classes.values())) == 1:
                a.append(1)
            else:
                best_class = max(a_values_for_classes, key=a_values_for_classes.get)
                a.append(0) if best_class == true_class_for_object else a.append(1)

        loo_result.update({h: np.sum(a)/length_of_all_objects})
        class1_errors = np.sum(a[:100])
        class2_errors = np.sum(a[100:])
        h += h_step

    end_time = time.process_time()
    print(end_time - start_time)
    return loo_result


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
        h.setdefault(cls, h_default)
        gammas.setdefault(cls, 0.0)

    for x in all_objects:

        for cl, class_objects in classes.items():
            l_y = len(class_objects)
            g = gamma(x, np.array(class_objects), h.get(cl), lambda_y, l_y, length_of_all_objects)
            a = {cl: g}
            a_values.update(a)
            gammas.update({cl: abs(g)})
        # убираем нулевые значения
        #a_values = {k: v for k, v in a_values.items() if v != 0.0}

        #if len(a_values) == 0:
        #    best_class = not_determined_class
        if len(a_values) > 1 and len(set(a_values.values())) == 1:
            best_class = not_determined_class
        else:
            # допущение, что класса всего два (хардкод)
            difference = gammas.get("1") - gammas.get("2")
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
n = 50
mu_2 = [10.0, 5.0]
std_2 = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float)

samples_1 = np.random.multivariate_normal(mu_1, std_1, n)
samples_2 = np.random.multivariate_normal(mu_2, std_2, n)

x1_y1 = samples_1[:, 0]
x2_y1 = samples_1[:, 1]

x1_y2 = samples_2[:, 0]
x2_y2 = samples_2[:, 1]
init_classes = {"1": samples_1.tolist(), "2": samples_2.tolist()}
# plot_samples(samples_1, samples_2)

# all_points = np.concatenate((samples_1, samples_2), axis=0)
all_points = get_points(-10, 20, -10, 20, 0.4)
# times = 10
# # while times > 0:
# classified_points, bound_points = classify(all_points, init_classes)
#
# class1 = np.array(classified_points["2"])
# x_bound, y_bound = find_boundary(class1[:, 0], class1[:, 1], 1)
#
# boundary = np.column_stack((x_bound, y_bound))
# #boundary = np.array(bound_points)
#
# not_determined = np.array(classified_points["not_determined"], dtype=float)
# plot_samples(np.array(classified_points["2"]), np.array(classified_points["1"]), not_determined, init_classes, boundary)

loo = leave_one_out(init_classes)
plt.plot(loo.keys(), loo.values(), lw=2, c='black')
plt.show()
