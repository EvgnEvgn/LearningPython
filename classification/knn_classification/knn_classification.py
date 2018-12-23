import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_frequency(data_):
    frequency = {}
    length = len(data_)
    for (key, value) in Counter(data_).items():
        frequency.update({str(key): round(value / length, 2)})

    return frequency


def mix_data(data_, coef=0.8):
    is_matched = False
    mixed = data_

    init_frequency = get_frequency(mixed.iloc[:, -1])

    while not is_matched:

        mixed = data_.sample(frac=1).reset_index(drop=True)
        amount_of_data = round(len(mixed) * coef)
        train = mixed.iloc[:amount_of_data, -1]

        frequency = get_frequency(train)

        is_differ = False
        for (key, value) in frequency.items():
            diff = abs(init_frequency[key] - value) / init_frequency[key]
            if diff > 0.01:
                is_differ = True
                break

        if not is_differ:
            return mixed


def classify(x_data_, y_data_):

    initial_data_frequency = get_frequency(y_data_)

    number_of_classes = len(initial_data_frequency.keys())

    train_x, test_x, train_y, test_y = train_test_split(x_data_, y_data_, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(train_x)

    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    frequency = get_frequency(sorted(train_y))

    model = KNeighborsClassifier(number_of_classes)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    if score > 0.89:
        print('score: {}'.format(score))
    predicted = model.predict(test_x)

    cross_table = pd.crosstab(test_y, predicted, rownames=['True Data'], colnames=['Predicted Data'])

    if score > 0.9:
        print(cross_table)
        print('Init freq = {0}'.format(collections.OrderedDict(sorted(initial_data_frequency.items()))))
        print('Train freq = {0}'.format(frequency))
        print(np.column_stack((test_x, predicted))[1:4])


def prepare_data(data_, need_to_mix=False):
    prepared_data = data_

    if need_to_mix:
        prepared_data = mix_data(prepared_data)

    return prepared_data


np.set_printoptions(threshold=np.nan)
data = pd.read_csv("ionosphere.data.txt", sep=',', dtype=float)

x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]

classify(x_data, y_data)

