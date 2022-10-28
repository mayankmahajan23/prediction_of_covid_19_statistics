import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("../data/ionosphere/ionosphere_data.csv")
data = np.array(x)
features = data[:,:-1]
target =  data[:,-1:]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
accuracy = []
train_accuracy = []
neighbors = []
initial_split = 0.3
final_split = 0.5
split_increment = 0.1

i = 5 # number of neighbors
while i <= 65 :
    neighbors.append(i)
    knn = KNeighborsClassifier(n_neighbors = i)

    j = initial_split
    result = []
    training_result = []
    while j <= final_split :
        X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = j, random_state = 0)

        knn.fit(X_train, y_train.ravel())
        result.append( knn.score(X_test, y_test) )
        training_result.append( knn.score(X_train, y_train) )

        j += split_increment
    i += 15
    accuracy.append(result)
    train_accuracy.append(training_result)

testFraction = []
j = initial_split
while j <= final_split :
    testFraction.append(j)
    j += split_increment

for i in range(len(neighbors)) :
    plt.plot(testFraction, accuracy[i], label = str(neighbors[i]) + 'testing data')
    plt.plot(testFraction, train_accuracy[i], label = str(neighbors[i]) + 'training data')
plt.legend()
plt.show()
plt.clf()
