import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from itertools import product
import math
import time
from numpy import sin, cos, pi
from scipy.optimize import leastsq
from sys import float_info
from scipy.interpolate import spline
from scipy.interpolate import interp1d


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


def get_ordered_list(points, x, y):
    #return sorted(points, key=lambda p: math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2))
    return sorted(points, key=lambda p: p[0] + p[1])
    #return points


def split_points(points):
    mean = np.mean(points, axis=0)
    left_points = []
    right_points = []

    for point in points:
        if point[0] <= mean[0]:
            left_points.append(point)
        else:
            right_points.append(point)

    return np.array(left_points), np.array(right_points)


def get_splited_boundary_points(boundary_pts):

    left_, right_ = split_points(boundary_pts)

    left_x = left_[:, 0]
    left_y = left_[:, 1]

    left_y_new = np.linspace(left_y.min(), left_y.max(), 1000)
    f = interp1d(left_y, left_x, kind='linear')
    left_x_new = f(left_y_new)

    right_x = right_[:, 0]
    right_y = right_[:, 1]
    right_y_new = np.linspace(right_y.min(), right_y.max(), 1000)
    f = interp1d(right_y, right_x, kind='linear')
    right_x_new = f(right_y_new)

    new_left = np.column_stack((left_x_new, left_y_new))
    new_right = np.column_stack((right_x_new, right_y_new))

    return new_left, new_right


def get_mean_by_normal_distribution(X):
    return np.mean(X, axis=0)


def get_std_by_normal_distribution(X, mean):
    a = np.subtract(X, mean)
    b = np.power(a, 2)

    c = np.mean(b, axis=0)

    return c


def std_inner(x, mean):
    a = np.subtract(x, mean)
    b = np.transpose(a)

    return np.dot(b, a)


def get_std_by_n_dim_normal_distribution(X, mean):
    res = [[0, 0], [0, 0]]
    m = len(X)

    for i in range(0, m):
        a = np.matrix(np.subtract(X[i], mean))
        b = np.transpose(a)
        c = np.dot(b, a)
        res = np.add(c, res)

    res = np.divide(res, m - 1)

    return res


def get_normal_density_distribution(x, mean, std):
    a = np.multiply(std, math.sqrt(2 * math.pi))
    b = np.power(np.subtract(x, mean), 2)
    c = np.multiply(np.power(std, 2), 2)
    d = np.divide(b, c)
    e = np.multiply(d, -1)
    f = np.exp(e)

    return np.divide(f, a)


def get_n_dim_normal_density_distribution(x, mean, std):
    n_ = len(x)
    det = np.linalg.det(std)
    inv_std = np.matrix(np.linalg.inv(std))

    a = math.sqrt(math.pow(2 * pi, n_) * det)
    b = np.matrix(np.subtract(x, mean))
    c = np.transpose(b)
    d = np.dot(b, inv_std)
    e = np.dot(d, c)
    f = np.multiply(e, -0.5)
    g = np.exp(f)

    res = np.divide(g, a)

    return res


def plot_samples(sample1, sample2, sample3=None, classes=None, boundary_points=None, title=None, filename=None):
    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(sample1[:, 0], sample1[:, 1], c='#3f89ff', marker="+", s=50)
    plt.scatter(sample2[:, 0], sample2[:, 1], c='#ff7c7c', marker="+", s=50)

    if sample3 is not None and len(sample3) > 0:
        plt.scatter(sample3[:, 0], sample3[:, 1], c='grey', marker="+", s=50)

    if classes is not None and len(classes) > 0:
        colors = ['red', 'blue', 'green', 'orange', 'yellow', 'black']
        i = 0
        for cl, points in classes.items():
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], c=colors[i], marker="o", s=20)
            i += 1

    if boundary_points is not None and len(boundary_points) > 0:
        if isinstance(boundary_points, tuple):
            plt.plot(boundary_points[0][:, 0], boundary_points[0][:, 1], c='black', lw=5)
            plt.plot(boundary_points[1][:, 0], boundary_points[1][:, 1], c='black', lw=5)
        else:
            plt.plot(boundary_points[:, 0], boundary_points[:, 1], c='black', lw=2)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_training_set(x, y, title):
    plt.xlabel('Param X')
    plt.ylabel('Param Y')
    plt.scatter(x[:, 0], x[:, 1], c='blue', marker="o", s=20)
    plt.scatter(y[:, 0], y[:, 1], c='red', marker="o", s=20)

    if title is not None:
        plt.title(title)
    plt.savefig('training_set.png')
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


def gamma_for_normal_classifier(x, lambda_y, l_y, l, mu, std):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    probability_estimates = get_normal_density_distribution(x, mu, std)

    if any(x == 0.0 for x in probability_estimates):
        return -float_info.max
    else:
        c = sum([math.log(pe) for pe in probability_estimates])

    return b + c


def gamma_for_plugin_classifier(x, lambda_y, l_y, l, mu, std):
    prior_probability = get_prior_probability(l_y, l)
    b = math.log(lambda_y * prior_probability)

    probability_estimates = get_n_dim_normal_density_distribution(x, mu, std)

    if any(x == 0.0 for x in probability_estimates):
        return -float_info.max
    else:
        c = sum([math.log(pe) for pe in probability_estimates])

    return b + c


def leave_one_out(classes):
    start_time = time.process_time()
    h = 0.4
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

            if len(a_values_for_classes) > 1 and len(set(a_values_for_classes.values())) == 1:
                a.append(1)
            else:
                best_class = max(a_values_for_classes, key=a_values_for_classes.get)
                a.append(0) if best_class == true_class_for_object else a.append(1)

        loo_result.update({h: np.sum(a) / length_of_all_objects})
        h += h_step

    end_time = time.process_time()
    print(end_time - start_time)
    return loo_result


def classify(all_objects, classes, type=1, h_by_loo=None):
    start_time = time.process_time()
    lambda_y = 1
    length_of_all_objects = len(all_objects)
    h_default = h_by_loo
    result_classes = {"not_determined": []}
    a_values = {}
    h = {}
    boundary_points = []
    gammas = {}
    not_determined_class = "not_determined"
    # точность для разности гамм
    epsilon = 0.2
    mu = {}
    std = {}

    for cls, objects in classes.items():
        result_classes.setdefault(cls, [])
        a_values.setdefault(cls, 0.0)
        h.setdefault(cls, h_default)
        gammas.setdefault(cls, 0.0)
        if type != 1:
            mu.update({cls: get_mean_by_normal_distribution(np.array(objects))})
            print('mu{1} = {0}'.format(mu.get(cls), type))
            if type == 2:
                std_ = get_std_by_normal_distribution(objects, mu.get(cls))
            else:
                std_ = get_std_by_n_dim_normal_distribution(np.array(objects), mu.get(cls))
            std.update({cls: std_})
            print('std{1} = {0}'.format(std_, type))

    for x in all_objects:

        for cl, class_objects in classes.items():
            l_y = len(class_objects)
            g = 0.0
            if type == 1:
                g = gamma(x, np.array(class_objects), h.get(cl), lambda_y, l_y, length_of_all_objects)
            elif type == 2:
                g = gamma_for_normal_classifier(x, lambda_y, l_y, length_of_all_objects, mu.get(cl), std.get(cl))
            else:
                g = gamma_for_plugin_classifier(x, lambda_y, l_y, length_of_all_objects, mu.get(cl), std.get(cl))
            a = {cl: g}
            a_values.update(a)
            gammas.update({cl: abs(g)})

        if len(a_values) > 1 and len(set(a_values.values())) == 1:
            best_class = not_determined_class
        else:
            # допущение, что класса всего два (хардкод)
            difference = gammas.get("1") - gammas.get("2")
            if abs(difference) <= epsilon:
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
std_1 = np.matrix([[2.0, 0.0], [0.0, 5.0]], dtype=float)
n = 300
mu_2 = [8.0, 6.0]
std_2 = np.matrix([[3.0, 0.0], [0.0, 3.0]], dtype=float)

samples_1 = np.random.multivariate_normal(mu_1, std_1, n)
samples_2 = np.random.multivariate_normal(mu_2, std_2, n)

x1_y1 = samples_1[:, 0]
x2_y1 = samples_1[:, 1]

x1_y2 = samples_2[:, 0]
x2_y2 = samples_2[:, 1]
init_classes = {"1": samples_1.tolist(), "2": samples_2.tolist()}
plot_training_set(samples_1, samples_2, title="Обучающая выборка")

loo = leave_one_out(init_classes)
loo_x = np.array(list(loo.keys()))
loo_y = np.array(list(loo.values()))
xnew = np.linspace(loo_x.min(), loo_x.max(), 300)
y_smooth = spline(loo_x, loo_y, xnew)
plt.plot(xnew, y_smooth, lw=2, c='black')
plt.ylabel("LOO")
plt.xlabel("h")
plt.savefig('loo.png')
plt.show()

all_points = get_points(-50, 50, -50, 50, 0.4)

h = min(loo, key=loo.get)
print(h)
classified_points, bounds = classify(all_points, init_classes, 1, h)

class1 = np.array(classified_points["2"])
x_bound, y_bound = find_boundary(class1[:, 0], class1[:, 1], 3)

boundary = np.column_stack((x_bound, y_bound))

not_determined = np.array(classified_points["not_determined"], dtype=float)
plot_samples(np.array(classified_points["2"]), np.array(classified_points["1"]),
             not_determined, init_classes, boundary_points=boundary, filename='result_of_naive_classification.png',
             title="\"Наивный\" байесовский классификатор")

classified_points, bounds = classify(all_points, init_classes, 2)

class1 = np.array(classified_points["1"])
x_bound, y_bound = find_boundary(class1[:, 0], class1[:, 1], 3)

boundary = np.column_stack((x_bound, y_bound))
bounds = np.array(bounds)

left, right = get_splited_boundary_points(bounds)

not_determined = np.array(classified_points["not_determined"], dtype=float)
class2 = classified_points["2"]
class1 = classified_points["1"]


plot_samples(np.array(class2), np.array(class1),
             not_determined, init_classes, boundary_points=(left, right), filename='result_of_normal_naive_classification.png',
             title="\"Наивный\" нормальный байесовский классификатор")

classified_points, bounds = classify(all_points, init_classes, 3)

left, right = get_splited_boundary_points(bounds)

not_determined = np.array(classified_points["not_determined"], dtype=float)
plot_samples(np.array(classified_points["2"]), np.array(classified_points["1"]),
             not_determined, init_classes, boundary_points=(left, right), filename='result_of_plugin_classification.png',
             title="Подстановочный классификатор (plug-in)")
