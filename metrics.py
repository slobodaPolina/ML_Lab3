from sklearn.metrics import f1_score


def f_score (y_pred, y_actual):
    return f1_score(y_actual, y_pred, average='binary', pos_label='P')