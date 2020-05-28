import pandas as pd
import datetime
import numpy as np

dati = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/manipulated_pun.csv',
                  sep = ";", decimal = ",", parse_dates = ['Data'], dtype = {'Ora': np.int16})

def from_number_to_hour(row):
    row = row - 1
    row = datetime.time(row).strftime('%H:00:00')
    return(row)

dati['Ora'] = dati['Ora'].apply(from_number_to_hour)

def merge_data_ora(x, y):
    row = str(x.date()) + ' ' + y
    return(row)

dati['Data'] = dati.apply(lambda x: merge_data_ora(x['Data'], x['Ora']), axis = 1)

dati.drop(columns = ['Ora'], inplace = True)

dati.to_csv('dati_PUN_17-20.csv', index = False, sep = ",")
