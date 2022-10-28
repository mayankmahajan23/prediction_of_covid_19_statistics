import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import msvcrt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.neural_network import MLPClassifier # multi-layer perceptron (MLP) algorithm that trains using Backpropagation
from sklearn.neural_network import MLPRegressor

def ann_regression (data, layer_size, xlabel, ylabel, title):
    start = 0
    end = len(data)
    extra_days = 15
    offset_july_31 = 16 + 31
    offset_june_15 = 1

    features = data[:, 1].reshape(len(data), 1).astype('int')
    target = data[:, 2].astype('float')
    X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = 0.25)

    clf = MLPRegressor(solver = 'lbfgs', alpha = 1e-7, hidden_layer_sizes = layer_size, random_state = 1, max_iter = 100000)
    clf.fit(X_train, y_train)

    print([coef.shape for coef in clf.coefs_], ' , ', clf.n_outputs_)
    print(clf.score(X_test, y_test))

    plt.plot(data[:, 1], data[:, 2], label = 'original data')
    x_axis = np.arange(0, len(data) + offset_july_31 + extra_days).reshape(len(data) + offset_july_31 + extra_days, 1)
    y_axis = clf.predict(x_axis)
    plt.plot(x_axis, y_axis, label = 'predicted data (plotted)', color = 'k')

    print(y_axis[end - start + offset_june_15 : end - start + offset_july_31])
    plt.scatter(len(data) + offset_july_31 - 1, y_axis[len(data) + offset_july_31 - 1], color = 'r', label = '31st July \'20')
    plt.scatter(end - start + offset_june_15, y_axis[end - start + offset_june_15], color = 'c', label = '15th June \'20')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + ': ann regression (network = ' + str(layer_size[0]) + '*' + str(layer_size[0]) + '*' + str(layer_size[0]) + ')')
    plt.legend()
    plt.show()
    plt.clf()
    print()

    return clf.score(X_test, y_test)
def ann_classification (data, layer_size, quant_step):
    start = 0
    end = len(data)
    extra_days = 15
    offset_july_31 = 16 + 31

    bins = np.arange(np.min(data[:,2]), np.max(data[:,2]) + quant_step, quant_step)
    quant_data = copy.deepcopy(data)
    quant_data[:,2] = np.digitize(data[:,2], bins, right = True)

    features = quant_data[:, 1].reshape(len(quant_data), 1).astype('int')
    target = quant_data[:, 2].astype('float')
    X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = 0.25)

    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = layer_size, random_state = 1, max_iter = 100000)

    clf.fit(X_train, y_train)

    print(len(np.unique(y_train)))
    print([coef.shape for coef in clf.coefs_], ' , ', clf.n_outputs_)
    print(clf.score(X_test, y_test))

    plt.plot(quant_data[:, 1], quant_data[:, 2], label = 'original data')
    x_axis = np.arange(0, len(quant_data) + offset_july_31 + extra_days).reshape(len(quant_data) + offset_july_31 + extra_days, 1)
    y_axis = clf.predict(x_axis)
    plt.plot(x_axis, y_axis, label = 'predicted data (plotted)')
    plt.scatter(x_axis, y_axis, label = 'predicted data (scattered)', color = 'k')
    plt.scatter(len(data) + offset_july_31 - 1, y_axis[len(data) + offset_july_31 - 1], color = 'r', label = '31st July \'20')
    plt.legend()
    plt.show()
    plt.clf()
    print()

    return clf.score(X_test, y_test)
def call_ann_regression (data, start, stop, step, flag, xlabel, ylabel, title):
    if (flag == 1):
        data[:, 2] = preprocessing.scale(data[:, 2]) # standardize

    score = []
    for i in range(start, stop, step):
        score.append((ann_regression(data, (i,i,i), xlabel, ylabel, title), i))
    score = np.array(score, dtype = [('score', 'float64'), ('nodes per layer', 'int')])
    score = np.sort(score, order = 'score')
    print(score)
    print()
def call_ann_classification (data, start, stop, step, quant_step):
    score = []
    for i in range(start, stop, step):
        score.append((ann_classification(data, (i,i,i), quant_step), i))
    score = np.array(score, dtype = [('score', 'float64'), ('nodes per layer', 'int')])
    score = np.sort(score, order = 'score')
    print(score)
    print()

def poly_reg (data, degree, xlabel, ylabel, title):
    start = 0
    end = len(data)
    extra_days = 15
    offset_july_31 = 35
    offset_june_15 = 1

    features = data[:, 1].reshape(len(data), 1) # because sklearn expects a 2d array
    target = data[:, 2]

    score = []
    reg = LinearRegression(fit_intercept = True, normalize = False)

    x_axis = np.arange(0, end - start + offset_july_31 + extra_days).reshape(end - start + offset_july_31 + extra_days, 1)
    for deg in range(1, degree + 1):
        poly = PolynomialFeatures(degree = deg)
        poly_features = poly.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split (poly_features, target, test_size = 0.25)

        reg.fit(X_train, y_train)
        result = reg.score(X_test, y_test)
        score.append((deg, result))

        weight = reg.coef_
        bias = reg.intercept_
        plt.plot(features, target, label = 'original data')

        poly_features = poly.fit_transform(x_axis)
        y_axis = np.dot(poly_features, weight) + bias
        plt.plot(x_axis, y_axis, label = 'predicted curve', color = 'k')

        print(y_axis[end - start + offset_june_15 : end - start + offset_july_31])
        plt.scatter(end - start + offset_july_31, y_axis[end - start + offset_july_31], color = 'r', label = '31st July \'20')
        plt.scatter(end - start + offset_june_15, y_axis[end - start + offset_june_15], color = 'c', label = '15th June \'20')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + ': polynomial regression (d = ' + str(deg) + ')')
        plt.legend()
        #plt.xticks(rotation = 90)
        plt.show()
        plt.clf()

    score = np.array(score, dtype = [('degree', 'int'), ('accuracy', 'float64')])
    score = np.sort(score, order = 'accuracy')
    print(score)
    print()

complete_data = pd.read_csv("../data/covid-19_data_india.csv")
print(complete_data)
complete_data = np.array(complete_data)

infections_by_date = []
deaths_by_date = []
closed_cases_by_date = []
old_date = complete_data[0,1]
infection_sum = 0
death_sum = 0
closed_cases = 0
i = 0
j = 0
for date in complete_data[:,1]:
    if (old_date == date):
        infection_sum += (complete_data[i,-1])
        death_sum += (complete_data[i,-2])
        closed_cases += (complete_data[i,-2]) + (complete_data[i,-3])
    else:
        infections_by_date.append((old_date, j, infection_sum))
        infection_sum = 0
        infection_sum += (complete_data[i,-1])

        deaths_by_date.append([old_date, j, death_sum])
        death_sum = 0
        death_sum += (complete_data[i,-2])

        closed_cases_by_date.append([old_date, j, closed_cases])
        closed_cases = 0
        closed_cases += (complete_data[i,-2]) + (complete_data[i,-3])

        old_date = date
        j += 1
    i += 1
infections_by_date.append((old_date, j, infection_sum))
deaths_by_date.append([old_date, j, death_sum])
closed_cases_by_date.append([old_date, j, closed_cases])
infections_by_date = np.array(infections_by_date, dtype = object)
deaths_by_date = np.array(deaths_by_date, dtype = object)
closed_cases_by_date = np.array(closed_cases_by_date, dtype = object)

import copy
death_rate_by_date = copy.deepcopy(deaths_by_date)
recovery_rate_by_date = copy.deepcopy(deaths_by_date)

# note: death_rate and recovery_rate are undefined for which closed_cases = 0
num_till_zero = 0
while (closed_cases_by_date[num_till_zero, 2] == 0):
    num_till_zero += 1
for i in range(num_till_zero, len(death_rate_by_date)):
    death_rate = float(deaths_by_date[i,2] / closed_cases_by_date[i, 2])
    recovery_rate = float((closed_cases_by_date[i, 2] - deaths_by_date[i,2]) / closed_cases_by_date[i, 2])
    death_rate_by_date[i,2] = death_rate
    recovery_rate_by_date[i,2] = recovery_rate

    death_rate_by_date[i,1] = i
    recovery_rate_by_date[i,1] = i

data_x_axis = np.arange(len(deaths_by_date) - num_till_zero)
death_rate_by_date = death_rate_by_date[num_till_zero : len(deaths_by_date)]
death_rate_by_date[:, 1] = data_x_axis
recovery_rate_by_date = recovery_rate_by_date[num_till_zero : len(deaths_by_date)]
recovery_rate_by_date[:, 1] = data_x_axis

infections_per_day = copy.deepcopy(deaths_by_date)
for i in range(len(deaths_by_date) - 1):
    infections_per_day[i + 1, 2] = infections_by_date[i + 1, 2] - infections_by_date[i, 2]
infection_rate_by_date = copy.deepcopy(infections_per_day)
infection_rate_by_date[:, 2] = infection_rate_by_date[:, 2] / 1380004385

# 1. infection rate: good analysis - degree 3 is the best, but ann was better in my opinion
#poly_reg(infection_rate_by_date, 5, 'numbers', 'infection rate', 'infection rate')
#call_ann_regression(infection_rate_by_date, 100, 401, 50, 1, 'numbers', 'infection rate', 'infection rate')

# 1. recovery rate: moderately good curve - gave one very good at 100, par ab nhi aa rha; ann_classification gives bad results for future
#call_ann_regression(recovery_rate_by_date, 100, 401, 50, 0, 'numbers', 'recovery rate', 'recovery rate')
#call_ann_classification(recovery_rate_by_date, 40, 160, 20, 0.001)




age_wise_cases = pd.read_csv('../data/35-50_india_cases.csv')
age_wise_deaths = pd.read_csv('../data/35-50_india_deaths.csv')
age_wise_cases = np.array(age_wise_cases)
age_wise_deaths = np.array(age_wise_deaths)

# http://statisticstimes.com/demographics/population-of-india.php - 273120352
infection_rate_by_date = copy.deepcopy(age_wise_cases[:, :-1])
infection_rate_by_date[:, 2] = infection_rate_by_date[:, 2] / 273120352

# 2. age_wise death_rate: can't calculate death rate from deaths and cases alone
    # so have to use the formula: death_rate = deaths / total infections
death_rate_by_date = copy.deepcopy(age_wise_deaths[:, :-2])
death_rate_by_date[:, 2] = age_wise_deaths[:, 3]
death_rate_by_date[:, 2] = death_rate_by_date[:, 2] / age_wise_deaths[:, -1]
#poly_reg(death_rate_by_date, 2, 'numbers', 'death rate', 'death rate (age 35-50)') # degree one is good
#call_ann_regression(death_rate_by_date, 5, 201, 20, 1, 'numbers', 'death rate', 'death rate (age 35-50)')

age_wise_deaths = pd.read_csv('../data/Provisional_COVID-19_Death_Counts_by_Sex_Age_and_State.csv')
age_wise_deaths = np.array(age_wise_deaths)

temp_data = []
for i in range(len(age_wise_deaths)):
    if (age_wise_deaths[i, 5] == '35-44 years'):
        temp_data.append(age_wise_deaths[i, :])
    elif (age_wise_deaths[i, 5] == '45-54 years'):
        temp_data.append(age_wise_deaths[i, :])
temp_data = np.array(temp_data, dtype = object)
age_wise_deaths = temp_data

# 3. age-wise infection rate: good analysis - no problems (many good curves)
#call_ann_regression(infection_rate_by_date, 5, 71, 10, 1, 'numbers', 'infection rate', 'infection rate (age 35-50)')




def active_cases_prediction (data, start, stop, step):
    active_cases_by_date = []
    old_date = data[0,1]
    infection_sum = 0
    closed_cases = 0
    active_cases = 0
    i = 0
    j = 0
    for date in data[:,1]:
        if (old_date == date):
            infection_sum += (data[i,-1])
            closed_cases += (data[i,-3]) + (data[i,-2])
            active_cases += infection_sum - closed_cases
        else:
            active_cases_by_date.append((old_date, j, active_cases))
            infection_sum = 0
            infection_sum += (data[i,-1])
            closed_cases = 0
            closed_cases += (data[i,-3]) + (data[i,-2])
            active_cases = 0
            active_cases += infection_sum + closed_cases

            old_date = date
            j += 1
        i += 1
    active_cases_by_date.append((old_date, j, active_cases))
    active_cases_by_date = np.array(active_cases_by_date, dtype = object)

    call_ann_regression(active_cases_by_date, start, stop, step, 0, 'numbers', 'active cases', 'italy active cases')

italy_data = pd.read_csv('../data/covid19_italy_region.csv')
italy_data = np.array(italy_data)
#active_cases_prediction(italy_data, 5, 51, 5)

print(infections_by_date)
poly_reg(infections_by_date, 5, 'numbers', 'infections', 'infections') # degree one is good
