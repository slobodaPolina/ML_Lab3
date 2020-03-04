from parameters import svc as svc_parameters
from svc import svc

results = []
chips = 'data/chips.csv'
geyser = 'data/geyser.csv'

for kernel in svc_parameters['kernel']:
    for regularization_strength in svc_parameters['regularization_strength']:
        avg_fscore, parameters = svc(chips, {'kernel': kernel, 'regularization_strength': regularization_strength})

        results.append({'avg_fscore': avg_fscore, 'parameters': parameters})

results = sorted(results, key=lambda k: k['avg_fscore'], reverse=True)

print(results)
print(results[0])
