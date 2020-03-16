import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from parameters import best_params


def colors(labels):
    return ['green' if label == 'P' else 'red' for label in labels]


# HERE we can switch the file
case = 'chips'
#case = 'geyser'

filename = best_params[case]['filename']
np_dataset = np.array(pd.read_csv(filename))

coordinates = np_dataset[:, :-1]  # x and y coordinates of the point
types = np_dataset[:, -1]  # answer (type for classification task)

# добавляем фичи на график - массив абсцисс, соответствующий массив ординат, массив соответствующих цветов точек
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors(types), cmap=plt.cm.Paired)

ax = plt.gca()
# границы по обеим осям
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# делим график на 30 частей по каждой стороне
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# хитрая функция, которая может размножить значения этих точек
YY, XX = np.meshgrid(yy, xx)
# получаем все возможные пары (сетка равномерно раскиданных по графику точек)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# учим модель имеющимися данными
model = svm.SVC(kernel=best_params[case]['kernel'], C=best_params[case]['regularization_strength'])
model.fit(coordinates, types)
# получаем решение модели и размножаем его для графка
Z = model.decision_function(xy).reshape(XX.shape)

# рисуем до данным координатам и решению модели линии на графике, 1 и 3 пунктиром, вторую сплошную
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# обводка для точек, которые не попали куда следует
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.show()
