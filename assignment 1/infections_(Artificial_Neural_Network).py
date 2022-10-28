import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # multi-layer perceptron (MLP) algorithm that trains using Backpropagation
from sklearn.neural_network import MLPRegressor
import msvcrt

complete_data = pd.read_csv("../data/covid-19_data_india.csv")

complete_data = np.array(complete_data)

infections_by_date = []
deaths_by_date = []
old_date = complete_data[0,1]
infection_sum = 0
death_sum = 0
i = 0
j = 0
for date in complete_data[:,1]:
    if (old_date == date):
        infection_sum += (complete_data[i,-1:])[0]
        death_sum += (complete_data[i,-2:])[0]
    else:
        infections_by_date.append((old_date, j, infection_sum))
        infection_sum = 0
        infection_sum += (complete_data[i,-1:])[0]

        deaths_by_date.append([old_date, j, death_sum])
        death_sum = 0
        death_sum += (complete_data[i,-2:])[0]

        old_date = date
        j += 1
    i += 1
infections_by_date.append((old_date, j, infection_sum))
deaths_by_date.append([old_date, j, death_sum])

infections_by_date = np.array(infections_by_date, dtype = object)
deaths_by_date = np.array(deaths_by_date, dtype = object)

# using the new data (sent by mridul mahindra)
data_by_date = pd.read_csv("../data/data_by_date.csv")
data_by_date = data_by_date.to_numpy()
infections_by_date = infections_by_date[16:120,:]
infections_by_date[:,2] = data_by_date[:,2]
infections_by_date[:,1] = np.arange(len(infections_by_date))
deaths_by_date = deaths_by_date[16:120,:]
deaths_by_date[:,2] = data_by_date[:,4]
deaths_by_date[:,1] = np.arange(len(deaths_by_date))

starting_point = 0
ending_point = len(infections_by_date) + 64 # july 31st
extra_days = 15  # (we plot more to see if the curve is valid near that area and doesn't overfit)

import copy
death_rate_by_date = copy.deepcopy(deaths_by_date) # copy.copy() means 'shallow copy'
recovery_rate_by_date = copy.deepcopy(deaths_by_date) # copy.deepcopy() means 'deep copy'
# The difference between shallow and deep copying is only relevant for compound objects (objects that contain other objects, like lists or class instances):
    # A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.
        # hence changes to the copied object's child members will be reflected in the original object
    # A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
        # hence changes to the copied object's child members will not be reflected in the original object
for i in range(len(deaths_by_date)):
    closed_cases = infections_by_date[i,2] - data_by_date[i,5]
    death_rate = float(deaths_by_date[i,2] / closed_cases)
    recovery_rate = float((closed_cases - deaths_by_date[i,2]) / closed_cases)
    death_rate_by_date[i,2] = death_rate
    recovery_rate_by_date[i,2] = recovery_rate

    death_rate_by_date[i,1] = i
    recovery_rate_by_date[i,1] = i

# cutting the irrelevant part of data to get better predictions
data_starting_point = 0
data_ending_point = len(infections_by_date)

infections_by_date = infections_by_date[data_starting_point : data_ending_point]
infections_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
deaths_by_date = deaths_by_date[data_starting_point : data_ending_point]
deaths_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
death_rate_by_date = death_rate_by_date[data_starting_point : data_ending_point]
death_rate_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
recovery_rate_by_date = recovery_rate_by_date[data_starting_point : data_ending_point]
recovery_rate_by_date[:,1] = np.arange(data_ending_point - data_starting_point)

'''
X = [[0.], [1.], [0.], [1.], [0.], [1.], [0.], [1.], ]
y = [0, 1, 0, 1, 0, 1, 0, 1]
X = np.array(X)
y = np.array(y)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2,4), random_state=1)
clf.fit(X, y)
print([coef.shape for coef in clf.coefs_], ' , ', clf.n_outputs_)

print(X)
print(y)
print(X.shape)
print(y.shape)
'''

def ann_classification (data, layer_size, quant_step):
    # first quantizing the data because we don't know how to do regression with neural networks
    bins = np.arange(0, np.max(data[:,2]) + quant_step, quant_step)
        # note: the last value in bins is not the maximum value (like we want). so even after setting right = True in digitize statement,
            # the maximum value will get the last index + 1, which is out of bounds. so add something to the stopping value like 0.1
    quant_data = copy.deepcopy(data)
    quant_data[:,2] = np.digitize(data[:,2], bins, right = True)
        # note: if right is 'false', the numbers will choose the quantized value smaller than them, and vice versa for 'true'
            # but if right is 'false', the number equal to the quantized value will choose the next quantized values' index.
            # hence for last quantized element, the index will be it's index + 1, which will be out of bounds
                # so choose 'true' if you have a value equal to max quantized level, false if by default
        # note: if the order of elements in bins is decreasing, right would refer to as the preceeding quantized level
            # in increasing order it refers to the next greater quantized level

    # dbt: if it accepts floating values as target, and hence we wouldn't need to divide some things by 100
    #for i in range(len(quant_data)):
        #quant_data[i,2] = bins[quant_data[i,2]]

    # note: the numpy array contains dtype = object, slicing it doesn't change the object. better to change it manually as follows, otherwise problems in future
    features = quant_data[:, 1].reshape(len(quant_data), 1).astype('int')
    target = quant_data[:, 2].astype('float')
    X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = 0.25)

    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = layer_size, random_state = 1, max_iter = 100000)
    # solver : {'lbfgs', 'sgd', 'adam'} (default = 'adam'); alphafloat: (default = 0.0001) L2 penalty (regularization term) parameter
    # hidden_layer_sizestuple, length = n_layers - 2, default=(100,). The ith element represents the number of neurons in the ith hidden layer.
        # '-2' because there is one input layer and one output layer. ex: (5, 2) two layers of sizes 5 and 2 respectively
    # activation: {'identity', 'logistic', 'tanh', 'relu'} (default = 'relu')
    # random_state: int or RandomState instance (default = None). Determines random number generation for weights and bias initialization
    # note: it supports only cross-entropy loss

    clf.fit(X_train, y_train)

    print(len(np.unique(y_train)))
    print([coef.shape for coef in clf.coefs_], ' , ', clf.n_outputs_)
    print(clf.score(X_test, y_test))

    plt.plot(data[:, 1], data[:, 2], label = 'original data')
    x_axis = np.arange(starting_point, ending_point + extra_days).reshape(ending_point + extra_days - starting_point, 1)
    y_axis = clf.predict(x_axis)
    plt.plot(x_axis, y_axis/100, label = 'predicted data (plotted)')
    plt.scatter(x_axis, y_axis/100, label = 'predicted data (scattered)', color = 'k')
    plt.scatter(ending_point - starting_point - 1, y_axis[ending_point - starting_point - 1], color = 'r', label = '31st July \'20')
    plt.legend()
    plt.show()
    plt.clf()
    print()

    return clf.score(X_test, y_test)
def ann_regression (data, layer_size):
    features = data[:, 1].reshape(len(data), 1).astype('int')
    target = data[:, 2].astype('float')
    X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = 0.25)

    clf = MLPRegressor(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = layer_size, random_state = 1, max_iter = 100000)
    clf.fit(X_train, y_train)

    print([coef.shape for coef in clf.coefs_], ' , ', clf.n_outputs_)
    print(clf.score(X_test, y_test))

    plt.plot(data[:, 1], data[:, 2], label = 'original data')
    x_axis = np.arange(starting_point, ending_point + extra_days).reshape(ending_point + extra_days - starting_point, 1)
    y_axis = clf.predict(x_axis)
    plt.plot(x_axis, y_axis, label = 'predicted data (plotted)')
    plt.scatter(x_axis, y_axis, label = 'predicted data (scattered)', color = 'k')
    plt.scatter(ending_point - starting_point - 1, y_axis[ending_point - starting_point - 1], color = 'r', label = '31st July \'20')
    plt.legend()
    plt.show()
    plt.clf()
    print()

    return clf.score(X_test, y_test)

score = []
for i in range(10, 71, 10):
    score.append((ann_regression(death_rate_by_date, (i,i,i)), i))
score = np.array(score, dtype = [('score', 'float64'), ('nodes per layer', 'int')])
score = np.sort(score, order = 'score')
print(score)
print()

# dbt: infections_by_date remaining
#ann_classification(infections_by_date, (4,5,2))
score = []
for i in range(10, 71, 10):
    score.append((ann_classification(death_rate_by_date, (i,i,i), 0.01), i))
score = np.array(score, dtype = [('score', 'float64'), ('nodes per layer', 'int')])
score = np.sort(score, order = 'score')
print(score)
print()

'''
y_pred = clf.predict(X_test)

X_test = X_test.sort()
y_test = y_test.sort()
print(X_test)
plt.plot(features, target)
plt.plot(X_test, y_test)
#plt.plot(X_test, y_pred)
plt.show()
plt.clf()
'''
