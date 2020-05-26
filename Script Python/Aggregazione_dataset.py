import pandas as pd

# Lettura dei datasets

anno_2004 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202004.csv',
                       sep = ";", usecols = [0, 1, 2], 
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2005 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202005.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2006 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202006.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2007 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202007.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2008 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202008.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2009 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202009.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2010 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202010.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2011 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202011.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2012 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202012.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2013 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202013.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2014 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202014.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2015 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202015.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2016 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202016.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2017 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202017.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2018 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202018.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2019 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202019.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

anno_2020 = pd.read_csv(r'https://raw.githubusercontent.com/mpichini1/DsLab/master/Datasets/Anno%202020.csv',
                       sep = ";", usecols = [0, 1, 2],
                       names = ['Data', 'Ora', 'PUN'], header = 0)

############# Aggregazione

frames = [anno_2004, anno_2005, anno_2006, anno_2007, anno_2008, anno_2009, anno_2010, 
         anno_2011, anno_2012, anno_2013, anno_2014, anno_2015, anno_2016, anno_2017,
         anno_2018, anno_2019, anno_2020]

dati_PUN_completi = pd.concat(frames, sort = False).reset_index() 
# Grazie a "sort = False" si mantiene l'ordine originale delle colonne.

dati_PUN_completi.drop(columns = 'index', inplace = True)

########### Creazione del csv

dati_PUN_completi.to_csv('dati_PUN_completi.csv', sep = ";") #Ho usato come sepratore il ;