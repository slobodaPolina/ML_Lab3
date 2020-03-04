folds = 4

svc = {
    'regularization_strength': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

best_params = {
    'chips': {
        'regularization_strength': 10.0,
        'kernel': 'rbf'
    },
    'geyser': {
        'regularization_strength': 10.0,
        'kernel': 'linear'
    }
}