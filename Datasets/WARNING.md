Per leggere il dataset *dati_PUN_completi*  si deve considerare che è strutturato secondo i canoni italiani. Quindi il separatore di campo è il ; mentre il separatore di decimali è la ,

Quindi se si usa pandas leggere il dataset in questo modo:



Se si usa python:

```python
dataset = pd.read_csv(r'dataset', sep = ";", decimal = ",", 
parse_dates = ['Data'])
```



Notare che parse__dates non è obbligatorio per leggere il dataset correttamente, ma consente di effettuare automaticamente il parsing della data.



Se si usa R:



```r
dataset <- read.csv('dataset', sep = ";", dec = ",")
```


