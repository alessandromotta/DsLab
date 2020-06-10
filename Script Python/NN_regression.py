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



# %%
df["Data"] = df["Data"].apply(lambda x: str(x))
df["Data"] = df["Data"].apply(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
df = df.loc[df['Data'] <= '2020-01-01']



# %%
df["Ora2"] = pd.to_datetime(df.Ora, unit="h").dt.strftime("%H:%M")
df["Datetime"] = df["Data"].dt.strftime("%Y-%m-%d") + " " + df["Ora2"]
df.index = pd.DatetimeIndex(df.Datetime)

# %%
df.head()

# %%
target = df["PUN"]
df["df24"] = target.shift(24)
target = df["PUN"].iloc[25:]
features = df[["PUN", "df24"]].iloc[25:]


# %%
df.head()
# %%
mask_test = df.index > "2019-12-25"
mask_train = df.index <= "2019-12-25"

# %%
test = df.loc[mask_test]
test

# %%
train = df.loc[mask_train]
train

# %%
train.dropna(inplace=True)


# %%
trainX = train['df24'][24:].values
testX = test['df24'].values
trainY = train['PUN'][24:].values
testY = test["PUN"].values

trainY


# %%
testY.shape


# %%
trainX.shape = (26089, 1)
testX.shape = (191, 1) 
trainY.shape = (26089, 1)
testY.shape = (191, 1)

# %%
model = tf.keras.Sequential()

model.add(keras.layers.Dense(64, activation='relu', input_shape=(trainX.shape[1],)))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4,
                                                  verbose=0, mode='auto')
model.fit(trainX, trainY, epochs=50, batch_size=1000, callbacks=[early_stopping])

# %%
predicted = model.predict(testX)


# %%
predicted_ = pd.DataFrame(predicted)
predicted_.head()


# %%
testY = pd.DataFrame(testY)
testY = testY.reset_index()

# %%

testY.head()

# %%

len(predicted)


# %%

results_vect = pd.concat([testY, predicted_], axis=1)
results_vect.columns = ['Datetime', 'Target', 'Reg']
results_vect 

# %%
predicted_final = predicted_.shift(-24)
len(predicted_final)

# %%

results_vect = pd.concat([testY, predicted_final], axis=1)

# %%
results_vect.columns = ['Datetime', 'Target', 'predicted_final']

# %%
results_vect.dropna(inplace=True)
results_vect[['Target', 'predicted_final']].plot()


# %%

results_vect['diff'] = abs(results_vect['Target'] - results_vect['predicted_final'])

print(results_vect['diff'].mean())

# %%

results_vect['diff'] = (abs(results_vect['Target'] - results_vect['predicted_final'])/results_vect['Target'])

print(results_vect['diff'].mean())
# %%
Hyperparameter optimization


%%
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
grid_result = grid.fit(trainX, trainY)

# %%
print(grid_result.best_score_)
print(grid_result.best_params_)

# %%
