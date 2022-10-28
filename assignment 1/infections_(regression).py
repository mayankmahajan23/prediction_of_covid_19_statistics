import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import msvcrt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # normalization
from sklearn.preprocessing import PolynomialFeatures

complete_data = pd.read_csv("../data/covid-19_data_india.csv")
print(complete_data)
# print(complete_data.isna().sum()) # checking missing values
print()

complete_data_latest = complete_data[complete_data['Date'] == complete_data['Date'][-1:].iloc[0]]
complete_data_latest = complete_data_latest.sort_values(by = ['Confirmed'], ascending = False)
plt.bar(complete_data_latest['State/UnionTerritory'][:5], complete_data_latest['Confirmed'][:5])
plt.ylabel('Number of Confirmed Cases', size = 12)
plt.title('States with maximum confirmed cases', size = 16)
#plt.show()
plt.clf()

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

# can convert to numpy here like this, but tried analyzing with pandas for a change
# infections_by_date = np.array(infections_by_date, dtype = object)
infections_by_date = pd.DataFrame(infections_by_date, columns = ['date', 'input', 'infections'])

plt.plot(infections_by_date[infections_by_date.columns[1]], infections_by_date[infections_by_date.columns[2]])
# note: if we give categorical data on either axis (like strings), the curve will always be a straight line, so give numerical data
    # here it works with dates on x-axix maybe because the data mentioned has common numeric index
#plt.show()
plt.clf()

#infections_by_date[infections_by_date.columns[2]] = np.log(infections_by_date[infections_by_date.columns[2]])
plt.plot(infections_by_date[infections_by_date.columns[1]], infections_by_date[infections_by_date.columns[2]])
#plt.show()
plt.clf()

infections_by_date = infections_by_date.to_numpy()
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

print(infections_by_date)
data_starting_point = 0
data_ending_point = len(infections_by_date)

starting_point = 0
ending_point = data_ending_point - data_starting_point + 143 # july 31st
extra_days = 15  # (we plot more to see if the curve is valid near that area and doesn't overfit)

def poly_reg (data, degree, xlabel, ylabel, title):
    features = data[:, 1].reshape(len(data), 1) # because sklearn expects a 2d array
    target = data[:, 2]

    score = []
    reg = LinearRegression(fit_intercept = True, normalize = True)
        # fit_intercept: bias or constant value (add a series of ones to the input data)
        # normalize : (This parameter is ignored when fit_intercept is set to False)
    x_axis = np.arange(starting_point, ending_point + extra_days).reshape(ending_point - starting_point + extra_days,1)
    for deg in range(1, degree + 1): # it is only good for degree = 1
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
        plt.plot(x_axis, y_axis, label = 'predicted curve')
        plt.scatter(data_ending_point - data_starting_point - 1 + 64, y_axis[data_ending_point - data_starting_point - 1 + 64], color = 'r', label = '31st July \'20')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + ': polynomial regression (d = ' + str(deg) + ')')
        plt.legend()
        plt.xticks(rotation = 90)
        plt.show()
        plt.clf()

    score = np.array(score, dtype = [('degree', 'int'), ('accuracy', 'float64')])
    score = np.sort(score, order = 'accuracy')
    print(score)
    print()

poly_reg(infections_by_date, 4, 'date', 'number of infections', 'total cases')

# using the new data (sent by mridul mahindra)
import copy
death_rate_by_date = copy.deepcopy(deaths_by_date) # copy.copy() means 'shallow copy'
recovery_rate_by_date = copy.deepcopy(deaths_by_date) # 'deep copy'
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
    print(closed_cases)

    death_rate_by_date[i,1] = i
    recovery_rate_by_date[i,1] = i

# cutting the irrelevant part of data to get better predictions
infections_by_date = infections_by_date[data_starting_point : data_ending_point]
infections_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
deaths_by_date = deaths_by_date[data_starting_point : data_ending_point]
deaths_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
death_rate_by_date = death_rate_by_date[data_starting_point : data_ending_point]
death_rate_by_date[:,1] = np.arange(data_ending_point - data_starting_point)
recovery_rate_by_date = recovery_rate_by_date[data_starting_point : data_ending_point]
recovery_rate_by_date[:,1] = np.arange(data_ending_point - data_starting_point)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True) # z = (x - u)/s
    # with_mean (x) (if false, then mean = 0), with_std (s) (if false, then taken as 1)
    # copy (if false, 'tries' to do 'in place' scaling (although its not guaranteed because returns a copy if if the data is not a NumPy array or scipy.sparse CSR matrix))
scaler.fit(infections_by_date[:,1:3]) # Compute the mean and std to be used for later scaling
# normalization. we are commenting this because can do later in regression classfier
#infections_by_date[:,1:3] = scaler.transform(infections_by_date[:,1:3]) # Perform standardization by centering and scaling
    # other functions - inverse_transform, fit_transform, get_params

poly_reg(infections_by_date, 4, 'date', 'number of infections', 'total cases')
#poly_reg(deaths_by_date, 7, 'date', 'deaths', 'total deaths')

#poly_reg(death_rate_by_date, 4, 'date', 'death rate', 'death rate')
#poly_reg(recovery_rate_by_date, 2, 'date', 'recovery rate', 'recovery rate')

''' # dbt: not working
death_rate_by_date = np.asarray(death_rate_by_date)
np.savetxt('death_rate_by_date.csv', death_rate_by_date, delimiter = ",")
'''

'''
infection = pd.DataFrame()
for date in complete_data['Date'].drop_duplicates(keep = 'first', inplace = False).values:
    temp_data = complete_data[complete_data['Date'] == date]
    sum = temp_data['Confirmed'].sum()
    temp_dict = {'date' : date, 'infections' : sum}
    infection.append(temp_dict, ignore_index = True)
print(infection)
'''
