# %%
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


# %%
df = pd.read_csv("manipulated_pun.csv", sep=";", decimal=',', header='infer')

df.head()
df.info()
# %%
df2 = pd.read_csv("sinusoidi.csv", sep=",", decimal='.', header='infer')

df2.info




# %%
df["Data"] = df["Data"].apply(lambda x: str(x))
df["Data"] = df["Data"].apply(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
df = df.loc[df['Data'] < '2020-01-01']

df.info()


# %%

df3 = pd.concat([df, df2], axis=1)

df3.head()


# %%

df3.drop(['Data', 'Ora'], axis=1, inplace=True)

df3.head()

# %%

df3.shape

# %%
x = df3.drop(['PUN'], axis=1)
y = df3['PUN']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

X_train = X_train.values

X_test = X_test.values

# %%

from sklearn.svm import SVR


# %%

svr_rbf = SVR(kernel='linear', C=1.5, epsilon=1e-07)


# %%
svr_rbf.fit(X_train, y_train)
y_rbf=svr_rbf.predict(X_test)
predicted=svr_rbf.predict(X_test)
predicted_ = pd.DataFrame(predicted)
predicted_.head()


# %%
y_test = y_test.reset_index()

# %%

y_test.head()

# %%

len(predicted)


# %%

results_vect = pd.concat([y_test, predicted_], axis=1)
results_vect.columns = ['Datetime', 'Target', 'Reg']

# %%
predicted_final = predicted_.shift(-168)
len(predicted_final)

# %%

results_vect = pd.concat([y_test, predicted_final], axis=1)

# %%
results_vect.columns = ['Datetime', 'Target', 'predicted_final']

# %%
results_vect[['Target', 'predicted_final']].plot()

# %%

from sklearn.model_selection import cross_val_score




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
parameters = {'kernel': ('linear', 'rbf'), 'C':[1.5, 10, 20, 50, 100, 200, 400 ],'gamma': [1e-7, 1e-4, 1e-2, 1e-1, 0.3, 0.4], 'epsilon': [0.1, 0.3, 1, 3, 5, 10]}
clf = SVR()  #inizialise the classifier
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=parameters,
                                   n_iter=n_iter_search, cv=5)

# %%

parameters
random_search.fit(X_train, y_train)
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
