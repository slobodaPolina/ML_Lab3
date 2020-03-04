import numpy as np


def cross_val_split (dataset, fold_qty, test_fold_num):
    folds = np.array_split(dataset, fold_qty)

    test_set = folds.pop(test_fold_num)
    train_set = np.concatenate(folds)

    return train_set, test_set
