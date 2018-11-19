import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random as rd
from collections import Counter
from sklearn.utils import shuffle


def get_frequency(data):
    frequency = {}
    length = len(data)
    for (key, value) in Counter(data).items():
        frequency.update({str(key): round(value / length, 2)})

    return frequency


def mix_data(data, coef=0.8):
    is_matched = False
    mixed = data

    init_frequency = get_frequency(mixed.iloc[:, -1])

    while not is_matched:

        mixed = data.sample(frac=1).reset_index(drop=True)
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


def classify(x_data, y_data):
    train_coef = 0.8

    initial_data_frequency = get_frequency(y_data)

    number_of_classes = len(initial_data_frequency.keys())

    amount_of_train_data = round(len(y_data) * train_coef)

    train_y = y_data.iloc[:amount_of_train_data]
    train_x = x_data.iloc[:amount_of_train_data, :]

    test_y = y_data.iloc[amount_of_train_data:]
    test_x = x_data.iloc[amount_of_train_data:, :]

    frequency = get_frequency(sorted(test_y))
    model = KNeighborsClassifier(number_of_classes)
    model.fit(train_x, train_y)

    print('score: {}'.format(model.score(test_x, test_y)))
    predicted = model.predict(test_x)

    crossTable = pd.crosstab(test_y, predicted, rownames=['True Data'], colnames=['Predicted Data'])
    print(crossTable)

def prepare_data(data, need_to_mix=False):
    prepared_data = data

    if need_to_mix:
        prepared_data = mix_data(prepared_data)

    return prepared_data


data = pd.read_csv("wisc_bc_data.csv", sep=',')
data = data.drop(['id'], axis=1)
data = prepare_data(data)

x_data = data.drop(['diagnosis'], axis=1)
y_data = data['diagnosis']


classify(x_data, y_data)

