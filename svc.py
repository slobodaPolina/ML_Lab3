import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from prepare_data import prepare_data, separate_features_labels
from cross_validation import cross_val_split
from parameters import folds
from parameters import svc as parameters
from metrics import f_score


def svc (filename, parameters):

    dataset, _, _ = prepare_data(filename, normalization=False)

    np.random.shuffle(dataset)

    model = SVC(C=parameters['regularization_strength'], kernel=parameters['kernel'])
    fscores = []

    print('SVC parameters: {}'.format(parameters))

    for fold in range(folds):
        cross_val_train, cross_val_test = cross_val_split(dataset, folds, fold)

        cross_val_train_features, cross_val_train_labels = separate_features_labels(cross_val_train)
        cross_val_test_features, cross_val_test_labels = separate_features_labels(cross_val_test)

        model.fit(cross_val_train_features, cross_val_train_labels)

        cross_val_predicted_labels = model.predict(cross_val_test_features)
        fscore = f_score(cross_val_predicted_labels, cross_val_test_labels)

        fscores.append(fscore)

        print('Fold: {}. Cross-validation F-score: {}'.format(fold+1, fscore))

    avg_fscore = np.mean(fscores)

    print('Average fscore: {}'.format(avg_fscore))

    return avg_fscore, parameters
