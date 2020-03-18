import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from parameters import svc as svc_parameters


# поделить массив данных на первые и последний столбец
def separate_features_labels(dataset):
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    return features, labels


# разделить датасет, чтобы тест-часть составила test_parts_amount / cross_validation_parts_amount часть исходного массива данных
def cross_val_split(dataset, cross_validation_parts_amount, test_parts_amount):
    # делим все данные (все точки) на несколько (cross_validation_parts_amount) массивов
    parts = np.array_split(dataset, cross_validation_parts_amount)
    # часть из них станет тестовой, оставшиеся - трейном
    test_set = parts.pop(test_parts_amount)
    train_set = np.concatenate(parts)
    return train_set, test_set


def svc(filename, parameters):
    np_dataset = np.array(pd.read_csv(filename))
    features, labels = separate_features_labels(np_dataset)
    # формируем массив из строк файла (x, y, type)
    dataset = np.concatenate((features, np.reshape(labels, (-1, 1))), axis=1)
    np.random.shuffle(dataset)
    model = SVC(C=parameters['regularization_strength'], kernel=parameters['kernel'], degree=parameters['degree'], gamma=parameters['gamma'])
    fscores = []

    for test_parts_amount in range(svc_parameters['cross_validation_parts']):
        # делим точки на обучающие и тестовые
        cross_val_train, cross_val_test = cross_val_split(dataset, svc_parameters['cross_validation_parts'], test_parts_amount)
        cross_val_train_features, cross_val_train_labels = separate_features_labels(cross_val_train)
        cross_val_test_features, cross_val_test_labels = separate_features_labels(cross_val_test)
        # кормим модель обучающими данными
        model.fit(cross_val_train_features, cross_val_train_labels)
        # смотрим, как отработает на тестовых
        cross_val_predicted_labels = model.predict(cross_val_test_features)
        # считаем f-меру (функцию оценки), сохраняем ее
        # report results for the class specified by pos_label
        fscores.append(f1_score(cross_val_test_labels, cross_val_predicted_labels, average='binary', pos_label='P'))
    return np.mean(fscores)


results = []
chips = 'data/chips.csv'
geyser = 'data/geyser.csv'
filename = geyser  # HERE we can change it

for kernel in svc_parameters['kernel']:
    for regularization_strength in svc_parameters['regularization_strength']:
        if kernel == 'linear':
            parameters = {'kernel': kernel, 'regularization_strength': regularization_strength, 'gamma': 'auto', 'degree': 1}  # fake params (they will not be used) not to fail in svc - do it better
            results.append({'avg_fscore': svc(filename, parameters), 'parameters': parameters})
        else:
            for gamma in svc_parameters['gamma']:
                if kernel == 'rbf' or kernel == 'sigmoid':
                    parameters = {'kernel': kernel, 'regularization_strength': regularization_strength, 'gamma': gamma, 'degree': 1}
                    results.append({'avg_fscore': svc(filename, parameters), 'parameters': parameters})
                else:
                    for degree in svc_parameters['degree']:
                        if kernel == 'poly':
                            parameters = {'kernel': kernel, 'regularization_strength': regularization_strength, 'gamma': gamma, 'degree': degree}
                            results.append({'avg_fscore': svc(filename, parameters), 'parameters': parameters})
results = sorted(results, key=lambda k: k['avg_fscore'], reverse=True)
print(results[0])
