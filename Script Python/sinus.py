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

X_train.shape

# %%

# %%
model = tf.keras.Sequential()

model.add(keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4,
                                                  verbose=0, mode='auto')
model.fit(X_train, y_train, epochs=400, batch_size=1024, callbacks=[early_stopping])

# %%
predicted = model.predict(X_test)

# %%
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

len(predicted_final)

# %%

results_vect = pd.concat([y_test, predicted_final], axis=1)

# %%
results_vect.columns = ['Datetime', 'Target', 'predicted_final']

# %%
results_vect[['Target', 'predicted_final']].plot()

# %%


# %%
# Hyperparameter optimization


# %%
def create_model():
    model = tf.keras.Sequential()

    model.add(keras.layers.Dense(64, activation='relu',
                           input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# %%
model = KerasRegressor(build_fn=create_model, verbose=0)

# %%
 batch_size = [1000, 10000, 5000, 2000]
 epochs = [ 14, 16, 24, 30, 40, 50]
 param_grid = dict(batch_size=batch_size, epochs=epochs)

# %%
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# %%
g#rid_result = grid.fit(X_train, y_train)

# %%
print(grid_result.best_score_)
print(grid_result.best_params_)

# %%
