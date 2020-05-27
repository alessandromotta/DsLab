# %%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# %%
# reading the complete dataset
df = pd.read_csv(r'dati_PUN_completi2.csv',
                 sep=",", usecols=[0, 1, 2],
                 names=['Data', 'Ora', 'PUN'], header=0, parse_dates=['Data'])

df.info()
df.head()

# %%

# plotting the whole period from 2004

df1 = df.groupby('Data', as_index=False).mean()

plt.figure(figsize=(24, 12))
sns.lineplot(x="Data", y="PUN",
             data=df1, label="PUN")
plt.title("Energy PUN Price Italy")
plt.legend()


# %%
# creating the subset of the three-years period 2017-2020
df = df.loc[df.index > 111792]

# deleting the row containing the 25th hour due to the change of solar time
df = df.loc[df['Ora'] != 25]
df.info()
df.head(10)

# %%

# to visualize the curve of the price we summarize the values for each day

dfplt = df.groupby('Data', as_index=False).mean()

df.head()

# %%

plt.figure(figsize=(24, 12))
sns.lineplot(x="Data", y="PUN",
             data=dfplt, label="PUN")
plt.title("Energy PUN Price Italy")
plt.legend()


# %%
# creating the new manipulated csv file
df.to_csv('manipulated_pun.csv',
          sep=";", decimal=',', index=False)
