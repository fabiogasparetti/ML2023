# -*- coding: utf-8 -*-
"""ML2023_03_dataset_outliers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bVaoiEdFf_6JBVTiRp5wbgGq4yjK2RF5
"""

"""
File: ML2023_03_dataset_outliers
Author: Fabio Gasparetti
Date: 2024-04-03

Description: Dataset outliers

See also: https://www.kaggle.com/code/hknaralasetty/weights-outlier-removal

"""

# Nel terminale attivare l'environment Anaconda e installare le librerie se occorre:
#
# source activate python3_11_7_uniroma3
# conda install numpy
# conda install pandas
# conda install matplotlib
# conda install seaborn

# Se non si impiega Anaconda:
#
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn


# Per Colab i suddetti comandi sono inutili

import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
import os # per il filesystem

# This is a simple dataset to start with. It contains only the height (inches)
# and weights (pounds) of 25,000 different humans of 18 years of age.
# This dataset can be used to build a model that can predict the heights or
# weights of a human.

df = pd.read_csv("https://raw.githubusercontent.com/noahgift/regression-concepts/master/height-weight-25k.csv")
df.head()

#data_path = r'ml2023/datasets/SOCR-HeightWeight.csv'
#df = pd.read_csv(data_path)
df.head()

df["Weight(KGs)"] = df["Weight-Pounds"] * .45
df.head()

"""Seaborn è una libreria per la visualizzazione dei dati basata su Matplotlib. Fornisce un'interfaccia di alto livello. Alcuni vantaggi rispetto a Matplotlib:

* Stile predefinito: è dotata di diversi temi e palette di colori integrati
* Funzionalità integrate: funzioni di visualizzazione avanzate non sono disponibili in Matplotlib, come i grafici a violino, le mappe di calore e le mappe a grappolo.
* Integrazione con Pandas: progettata per funzionare  con Pandas DataFrames.
"""

# l'interfaccia con la libreria è simile a matplotlib:

# create a dataset
data = [1, 2, 3, 4, 5]

# create a histogram using seaborn
sns.histplot(data)

# add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Using Seaborn')

# show the plot
plt.show()

# create a histogram using matplotlib
plt.hist(data)

# add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Using Matplotlib')

# show the plot
plt.show()

# A histogram is a classic visualization tool that represents the
# distribution of one or more variables by counting the number of
# observations that fall within discrete bins.

# kde - If True, compute a kernel density estimate to smooth the distribution
# and show on the plot as (one or more) line(s).

sns.histplot(data = df, x = 'Weight(KGs)', kde = True)

# Draw a box plot to show distributions with respect to categories.

# Un box plot (o box-and-whisker plot) mostra la distribuzione dei dati
# quantitativi in modo da facilitare i confronti tra variabili o tra livelli
# di una variabile categorica.

# I quartili sono quei valori/modalità che ripartiscono la popolazione
# in quattro parti di uguale numerosità.
# Il box blu (IQR) rappresenta il 1º e il 3º quartile (Q1 e Q3),
# che equivalgono al 25º e al 75º percentile.
# La riga all'interno della scatola rappresenta il 2º quartile,
# che costituisce la mediana, cioè il valore (o indice del valore)
# che si trova nel mezzo della distribuzione.

# L'intervallo tra quartili (IQR), da cui prende il nome questo metodo di
# rilevamento degli outlier, è l'intervallo tra il primo e il terzo quartile
# (i bordi della scatola).

# Il grafo mostra i quartili del set di dati, mentre i whiskers (baffi)
# si estendono  per mostrare il resto della distribuzione, ad eccezione dei
# punti che sono stati determinati come "outlier" utilizzando un metodo che
# è una funzione dell'intervallo interquartile.

# Si considera anomalo (seguendo l'approccio John Tukey) qualsiasi datapoint che
# si allontani dal primo o dal terzo quartile di più di 1,5*IQR
# rispettivamente verso sx o verso dx.
# Corrispondono ai 2 baffi più esterni.

# In un diagramma a scatola e baffi classico, i baffi si estendono fino
# all'ultimo datapoint che non si considera "esterno".

sns.boxplot(data = df, x = 'Weight(KGs)')

titanic = sns.load_dataset('titanic')

titanic.head()

sns.boxplot(x=titanic["age"])

sns.boxplot(data=titanic, x="age", y="class")

sns.boxplot(data=titanic, x="class", y="age", hue="alive")

sns.boxplot(data=titanic, x="age", y="deck", whis=(0, 100))

survivors_details=titanic[titanic['survived']==1]
survivors_details

titanic['class'].value_counts()

#to find each class's highest fare
titanic.groupby('class')['fare'].max()

#to find average age of passengers by class
titanic.groupby('class')['age'].mean()

titanic=titanic.sort_values(by=['fare','age'])
titanic

"""Torniamo al dataset SOCR-HeightWeight"""

# ricaviamo altre statistiche di interesse
mean = df['Weight(KGs)'].mean()
print ('mean: ',mean)

standard_deviation = df['Weight(KGs)'].std()
print ('mean: ',standard_deviation)

lower_cut, upper_cut = mean -2 * standard_deviation , mean +2 * standard_deviation
print ('lower_cut', lower_cut)
print ('upper_cut', upper_cut)

# approccio Python/Dataframe per filtrare i dati in base al valore di una
# colonna all'interno di uno specifico intervallo
# boolean indexing

# data[(data.A>interval_left) & (data.A<interval_right)]

# approcci alternativi:
# -> data[data.A.between(interval.left+1, interval.right)]
# -> data.query('1 <= A <= 4')
# -> data[data.A.isin(range(1, 5))]
# -> Python lambda functions

df_clean = df[(df['Weight(KGs)'] > lower_cut) & (df['Weight(KGs)'] < upper_cut)]
print ('df_clean.head()',df_clean.head())

df.shape, df_clean.shape

sns.histplot(data = df_clean, x = 'Weight(KGs)', kde = True)

# Il grafico a violino è molto simile al box-and-whisker.
# Mostra la distribuzione dei punti dopo il raggruppamento per una (o più)
# variabili.

# A differenza di un box plot, il grafico mostra luna curva sottoposta a
# smoothing (KDE) che indica la densità di probabilità relativo ad uno specifico
# valore x


sns.violinplot(data = df_clean, x='Weight(KGs)')

# outlier removal using zscore

# Sebbene lo Z-Score sia un metodo molto efficiente per individuare e rimuovere
# gli outlier, ma è adatto solo per dati distribuiti normalmente
# Per gli altri dati conviene l'Inter quartile range (IQR).

df['zscore'] = (df['Weight(KGs)'] - df['Weight(KGs)'].mean()) / standard_deviation

df_clean2 = df[(df.zscore > -2) & (df.zscore < 2)]
df_clean2.shape

sns.boxplot(data = df_clean2, x='Weight(KGs)')