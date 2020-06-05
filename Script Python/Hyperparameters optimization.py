# %%
import pandas as pd
import numpy as np
import os

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform

# %%
df = pd.read_csv("manipulated_pun.csv", sep=";", decimal=',', header='infer')

df.head()
# %%
def data():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("manipulated_pun.csv", sep=";", decimal=',', header='infer')
    df = df.drop(['Data'], axis =1)
    train, test = train_test_split(df, test_size=0.2)
    y_train = train['PUN']
    X_train = train.drop(['PUN'], axis=1)
    y_test = test['PUN']
    X_test = test.drop(['PUN'], axis=1)
    X_train = X_train.values
    X_test = X_test.values
    return X_train, y_train, X_test, y_test

# %%

def create_model(X_train, y_train, X_test, y_test):
    import tensorflow.python.keras
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Activation, Dropout
    from tensorflow.python.keras import backend as K

    from tensorflow.python.keras import utils, layers, Sequential
    model = Sequential()
    # the model will take as input arrays of shape (*, X_train.shape[1])
    # and output arrays of shape (*, 25)
    model.add(Dense(25, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='mse', metrics=['mae'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(X_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation mae of the training epochs
    validation_mae = np.amax(result.history['mae']) 
    print('Best validation mae of epoch:', validation_mae)
    return {'loss': validation_mae, 'status': STATUS_OK, 'model': model}# %%


# %%
def create_model():
    model = tf.keras.Sequential()

    model.add(keras.layers.Dense(64, activation='relu',
                           input_shape=(trainX.shape[1],)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# %%
best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials(),
                                      notebook_name='Hyperparameters optimization.py')
# %%

print("Best performing model chosen hyper-parameters:")
print(best_run)