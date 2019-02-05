import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style 
from collections import Counter
style.use('fivethirtyeight')


dataset = {'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]] }
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distane = sqrt((features[0]-predict[0])**2 + (features[1] - predict[1])**2) # this is not fast!! because it looks at every datapoint also 
                                                                                                    #what if we had more than 2 dimensions
            #euclidean_distane = np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) #this will work but there is a simpler np way of doing it
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) # no longer looks like the euclidean formula but it's a faster highlevel function
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

results = k_nearest_neighbors(dataset, new_features, k=3)
print(results)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=results)
plt.show()