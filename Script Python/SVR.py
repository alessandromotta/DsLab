# %%
import datetime

import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras




# %%
df = pd.read_csv("manipulated_pun.csv", sep=";", decimal=',', header='infer')

df.head()

# Create a minimum and maximum processor object

# %%
df["Data"] = df["Data"].apply(lambda x: str(x))
df["Data"] = df["Data"].apply(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
df = df.loc[df['Data'] < '2020-01-01']


# %%
df2 = pd.read_csv("sinusoidi.csv", sep=",", header='infer')

df3 = pd.concat([df, df2], axis=1)

df3

# %%
df3["Ora2"] = pd.to_datetime(df3.Ora, unit="h").dt.strftime("%H:%M")
df3["Datetime"] = df3["Data"].dt.strftime("%Y-%m-%d") + " " + df3["Ora2"]
df3.index = pd.DatetimeIndex(df3.Datetime)

# %%
df3.head()

# %%

df3.drop(['Data', 'Ora', 'Ora2', 'Datetime'], axis=1, inplace=True)

df3.head()
# %%
target = df3["PUN"]
df3["df24"] = target.shift(24)
target = df3["PUN"].iloc[25:]
features = df3[["PUN", "df24"]].iloc[25:]


# %%
df3.head()
# %%
mask_test = df3.index > "2019-12-24"
mask_train = df3.index <= "2019-12-24"

# %%
test = df3.loc[mask_test]
test

# %%
train = df3.loc[mask_train]
train



# %%
X_train = train.drop("PUN", axis=1).values
X_train = X_train[24:]

# %%
X_test = test.drop("PUN", axis=1).values

# %%
y_train = train['PUN'][24:].values

# %%
y_test = test["PUN"].values

y_train


# %%
y_test.shape


# %%
y_train.shape = (26065, 1)
y_test.shape = (191, 1)


# %%

from sklearn.svm import SVR


# %%

svr_rbf = SVR(kernel = 'linear', C=1.5)


# %%
svr_rbf.fit(X_train, y_train)

predicted=svr_rbf.predict(X_test)
predicted_ = pd.DataFrame(predicted)
predicted_.head()


# %%

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index()

# %%

y_test.head()

# %%

len(predicted)


# %%

results_vect = pd.concat([y_test, predicted_], axis=1)
results_vect.columns = ['Datetime', 'Target', 'Reg']

# %%
predicted_final = predicted_.shift(-24)
len(predicted_final)

# %%

results_vect = pd.concat([y_test, predicted_final], axis=1)

# %%
results_vect.columns = ['Datetime', 'Target', 'predicted_final']

results_vect['Target'].mean()

# %%

results_vect.dropna(inplace=True)
results_vect[['Target', 'predicted_final']].plot()

# %%

# plt.plot(results_vect[['Target', 'predicted_final']])
# plt.savefig('svrocco.png')

# %% 
# X_train = X_train.values
# y_train=y_train.values

#svc_pred = svr_rbf.predict(X_train)

# plt.figure()
# plt.plot(y_train, color = "red")
# plt.plot(svc_pred)

# np.mean(np.abs(((svc_pred-y_train)/y_train)))

# %%
results_vect['diff'] = (abs(results_vect['Target'] - results_vect['predicted_final'])/results_vect['Target'])

results_vect['diff'].mean()

# %%

results_vect['diff'] = abs(results_vect['Target'] - results_vect['predicted_final'])

results_vect['diff'].mean()

# %%

from sklearn.model_selection import cross_val_score

y_test=test[["PUN_n"]]

# %% 

# %%
scores = cross_val_score(svr_rbf, X_test, y_test, scoring='neg_mean_absolute_error', cv=5)
scores

# %%

sorted(sklearn.metrics.SCORERS.keys())


# %%
np.mean(scores)


# %%
import matplotlib.pyplot as plt

epsilonvar = [0.1, 3, 10]
colors = ['r', 'b', 'g']

for i in range(len(epsilonvar)): 
    svr_rbf = SVR(kernel='rbf', C=1000, epsilon=epsilonvar[i])
    svr_rbf.fit(X_train, y_train)
    y_rbf = svr_rbf.predict(X_test)
    plt.scatter(X_test, y_test, color='darkorange', label='data')
    plt.scatter(X_test, y_rbf, color=colors[i], label='RBF model_epsilon ' + str(epsilonvar[i]))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

# %%

import matplotlib.pyplot as plt

Cvar = [0.1, 10, 1000]
colors = ['r', 'b', 'g']

for i in range(len(Cvar)): 
    svr_rbf = SVR(kernel='rbf', C=Cvar[i], gamma=0.1, epsilon=3)
    svr_rbf.fit(X_train, y_train)
    y_rbf = svr_rbf.predict(X_test)
    plt.scatter(X_test, y_test, color='darkorange', label='data')
    plt.scatter(X_test, y_rbf, color=colors[i], label='RBF model_' + str(Cvar[i]))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

# %%
from sklearn.svm import LinearSVR
svr_lin = LinearSVR(random_state=0, tol=1e-5, max_iter=1000000)
svr_lin.fit(X_train, y_train)
y_lin = svr_lin.predict(X_test)

# %%

scores = cross_val_score(svr_lin, X_test, y_test, scoring='neg_mean_absolute_error', cv=5)
scores

np.mean(scores)


# %%

import matplotlib.pyplot as plt

svr_lin = LinearSVR(random_state=0, tol=1e-5, max_iter=1000000)
svr_lin.fit(X_train, y_train)
y_rbf = svr_lin.predict(X_test)
plt.scatter(X_test, y_test, color='darkorange', label='data')
plt.scatter(X_test, y_rbf, color=colors[i], label='Linear model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# %%
from sklearn.svm import SVR
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C':[1.5, 10 ]}
clf = SVR()  # inizialise the classifier
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=parameters,
                                   n_iter=n_iter_search, cv=5)

# %%

parameters
random_search.fit(X_train, y_train)
# %%
print(random_search.best_estimator_)


# %%
from sklearn.svm import SVR
grid_param = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4]}
clf = SVR() #inizialise the classifier
gd_sr = GridSearchCV(estimator=clf,  
                     param_grid=grid_param,
                     scoring='neg_mean_absolute_error',
                     cv=5)

# %%
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_  
print(best_parameters)
