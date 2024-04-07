"""
File: ML2023_01_numpy_pandas.py
Author: Fabio Gasparetti
Date: 2024-04-03

Description: NumPy & Pandas 
"""

# Nel terminale attivare l'environment Anaconda e installare le librerie se occorre:
#
# source activate python3_11_7_uniroma3
# conda install numpy
# conda install pandas

# Se non si impiega Anaconda:
#
# pip install pandas
# pip install numpy

# Per Colab i suddetti comandi sono inutili

import numpy as np
import pandas as pd
import os # per il filesystem

# Gli  NumPy sono  più flessibili dei normali elenchi di Python. 
# Si chiamano ndarray perché possono avere un numero qualsiasi (n) di dimensioni (d). 
# Possono contenere un insieme di elementi di uno stesso tipo di dati 
# e possono essere vettori (monodimensionali) o matrici (multidimensionali). 
# Gli array di NumPy consentono un accesso rapido agli elementi e una manipolazione efficiente dei dati.

# Inizializziamo una lista Python - tipo di dato base
list1 = [1,2,3,4]
print('list1: ', list1)

# Convertiamo la lista in un ndarray 1-dimensionale, con 1 riga e 4 colonne
array1 = np.array(list1)
print('array1: ',array1)

# Per ottenere un ndarray a 2 dimensioni, è sufficiente passare una lista di liste
list2 = [[1,2,3],[4,5,6]]
array2 = np.array(list2)
print('array2: ',array2)


# Vediamo alcune operazioni utili per manipolare i dataset

# Durante la standardizzazizone "centriamo" i valori intorno allo 0
# tipicamente sottraendo il valore medio.
toyPrices = np.array([5,8,3,6])
mean = np.mean(toyPrices)
print ('mean: ',mean)
print ('toyPrices - mean: ',toyPrices - mean)

# il codice alternativo sulle liste è il seguente:
toyPrices = [5,8,3,6]
# print(toyPrices - 2) -- Not possible. Causes an error
for i in range(len(toyPrices)):
    toyPrices[i] -= 2
print(toyPrices)

# il parametro axis indica su quale asse elaborare il calcolo
# list2 ha dimensione 2x3
# axis=0 indica che il calcolo sarà fatto iterando ogni riga (e considerando tutti i valori singolarmente per colonna)
# se si omette axis, la struttura è "flattened" e il calcolo è esteso a tutti i valori
mean0 = np.mean(list2, axis=0)
mean1 = np.mean(list2, axis=1)
print ('mean0: ', mean0)
print ('mean1: ', mean1)


# Creiamo una Series con un NumPy array e un sistema di indicizzazione di default
ages = np.array([13,25,19])
series1 = pd.Series(ages)
print ('series1:\n', series1)

# Ora usiamo una indicizzazione basata su stringhe
ages = np.array([13,25,19])
series1 = pd.Series(ages,index=['Emma', 'Swetha', 'Serajh'])
print ('series1:\n', series1)

# Creiamo un DataFrame utilizzando un elenco di liste in Python. 
# Ogni elenco annidato rappresenta i dati di una riga del DataFrame. 
# Utilizziamo la parola chiave columns per passare l'elenco dei nomi 
# delle nostre colonne personalizzate.
dataf = pd.DataFrame([
    ['John Smith','123 Main St',34],
    ['Jane Doe', '456 Maple Ave',28],
    ['Joe Schmo', '789 Broadway',51]
    ],
    columns=['name','address','age'])
print ('dataf:\n',dataf)

# sostituisce l'indice di default con i valori di una colonna
dataf.set_index('name')

# Inizializziamo il DataFrame con un dizionario
dataf = pd.DataFrame({'name':['John Smith','Jane Doe','Joe Schmo'],
                      'address':['123 Main St','456 Maple Ave','789 Broadway'],
                      'age':[34,28,51]})
print ('dataf:\n',dataf)


# uso r prima della stringa per evitare l'escaping dei backslashes 
data_path = r'ml2023/datasets/SOCR-HeightWeight.csv'

df = pd.read_csv(data_path)
print('head(csv):\n',df.head())

