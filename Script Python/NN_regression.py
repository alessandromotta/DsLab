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
# %%

# Create x, where x the 'scores' column's values as floats
x = df[['PUN']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

df_normalized = pd.DataFrame(x_scaled)

df['PUN_n'] = df_normalized.values

# %%

df


# %%
df["Data"] = df["Data"].apply(lambda x: str(x))
df["Data"] = df["Data"].apply(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
df = df.loc[df['Data'] < '2020-01-01']


# %%


# %%
df["Ora2"] = pd.to_datetime(df.Ora, unit="h").dt.strftime("%H:%M")
df["Datetime"] = df["Data"].dt.strftime("%Y-%m-%d") + " " + df["Ora2"]
df.index = pd.DatetimeIndex(df.Datetime)

# %%
df.head()

# %%
target = df["PUN_n"]
df["df24"] = target.shift(168)
target = df["PUN_n"].iloc[169:]
features = df[["PUN_n", "df24"]].iloc[169:]


# %%
features

# %%
mask_test = df["Datetime"] > "2019-01-01"
mask_train = df["Datetime"] <= "2019-01-01"

# %%
test = df.loc[mask_test]
test

# %%
train = df.loc[mask_train]
train

# %%
trainX = train[["df24"]][168:]
testX = test[["df24"]]
trainY = train[["PUN_n"]][168:]
testY = test[["PUN_n"]]

trainY


# %%
trainX.shape


# %%
testX.shape

# %%
model = tf.keras.Sequential()

model.add(keras.layers.Dense(64, activation='relu', input_shape=(trainX.shape[1],)))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4,
                                                  verbose=0, mode='auto')
model.fit(trainX, trainY, epochs=400, batch_size=1024, callbacks=[early_stopping])

# %%
predicted = model.predict(testX)

# %%
predicted_ = pd.DataFrame(predicted)
predicted_.head()

# %%
testY = testY.reset_index()

# %%

testY.head()

# %%

len(predicted)

# %%

results_vect = pd.concat([testY, predicted_], axis=1)
results_vect.columns = ['Datetime', 'Target', 'Reg']

# %%
predicted_final = predicted_.shift(-24)
len(predicted_final)

# %%

results_vect = pd.concat([testY, predicted_final], axis=1)

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
                           input_shape=(trainX.shape[1],)))
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
g#rid_result = grid.fit(trainX, trainY)

# %%
print(grid_result.best_score_)
print(grid_result.best_params_)

# %%
