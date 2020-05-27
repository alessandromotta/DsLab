# %%
import pandas as pd

# %%
# reading the complete dataset
df = pd.read_csv(r'dati_PUN_completi2.csv',
                 sep=",", usecols=[0, 1, 2],
                 names=['Data', 'Ora', 'PUN'], header=0)

df.info()


# %%
# creating the subset of the three-years period 2017-2020
df = df.loc[df.index > 111792]

# deleting the row containing the 25th hour due to the change of solar time
df = df.loc[df['Ora'] != 25]
df.info()
df.head(10)


# %%
#creating the new manipulated csv file
df['PUN'] = df['PUN'].astype(float)

df.to_csv('manipulated_pun.csv',
          sep=";", decimal=',', index=False)

# %%
