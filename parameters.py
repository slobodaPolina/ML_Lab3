# Параметры для подбора лучшей комбинации
svc = {
    'cross_validation_parts': 5,
    'regularization_strength': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # rbf is gaussian actually
    'degree': [1, 2, 3, 4, 5],  # 'poly'
    'gamma': ['scale', 'auto']  # for ‘rbf’, ‘poly’ and ‘sigmoid’.
}

# Подобранные лучшие параметры
best_params = {
    'chips': {
        'regularization_strength': 10.0,
        'kernel': 'rbf',
        'gamma': 'auto',
        'degree': 3,
        'filename': 'data/chips.csv'
    },
    'geyser': {
        'regularization_strength': 10.0,
        'kernel': 'poly',
        'gamma': 'scale',
        'degree': 3,
        'filename': 'data/geyser.csv'
    }
}
