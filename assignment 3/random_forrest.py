import numpy as np # numerical python (scientific computing)
import pandas as pd # for data manipulation and analysis
import matplotlib.pyplot as plt # for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import msvcrt

x = pd.read_csv("../data/ionosphere/ionosphere_data.csv")
data = np.array(x)
features = data[:,:-1]
target =  data[:,-1:]

initial_estimators = 10
final_estimators = 810
estimator_increment = 200

initial_split = 0.3
final_split = 0.5
split_increment = 0.1

split_history = []
split = initial_split
while split <= final_split :
    split_history.append(split)
    split += split_increment

n_estimators = initial_estimators
while n_estimators <= final_estimators :
    score_history = []
    training_score = []
    split = initial_split
    while split <= final_split :
        X_train, X_test, y_train, y_test = train_test_split (features, target, random_state = 0, test_size = split)
        clf = RandomForestClassifier(n_estimators = n_estimators)
        clf = clf.fit(X_train, y_train.ravel())

        y_prediction = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_prediction)
        score_history.append(score)
        y_prediction = clf.predict(X_train)
        score = metrics.accuracy_score(y_train, y_prediction)
        training_score.append(score)

        split += split_increment

    plt.plot(split_history, score_history, label = str(n_estimators) + ' (testing data)')
    plt.plot(split_history, training_score, label = str(n_estimators) + ' (training data)')
    n_estimators += estimator_increment

plt.legend()
plt.show()
