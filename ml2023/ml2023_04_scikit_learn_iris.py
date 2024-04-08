# -*- coding: utf-8 -*-
"""ML2023_04_scikit-learn_iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12VjpK9T_p6ut3H8Fyr4gBvx3zxT1Ujk_
"""

"""
File: ML2023_04_scikit-learn_iris
Author: Fabio Gasparetti
Date: 2024-04-03

Description: Classificatore basato su percettrone e libreria scikit-learn

"""

# Nel terminale attivare l'environment Anaconda e installare le librerie se occorre:
#
# source activate python3_11_7_uniroma3
# conda install numpy
# conda install pandas
# conda install matplotlib
# conda install sklearn
# conda install mlxtend

# Se non si impiega Anaconda:
#
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install sklearn
# pip install mlxtend


# Per Colab i suddetti comandi sono inutili

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import sys

from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))

# Splitting data into 70% training and 30% test data:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# bincount: Count number of occurrences of each value in array of non-negative ints
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

X_train[:10]

# Standardizing the features:

# StandardScaler: Standardize features by removing the mean and scaling to unit variance.

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std[:10]

# ## Training a perceptron via scikit-learn

# Il costruttore indica l'algoritmo di ML che si intende impiegare
# Per una lista completa: https://scikit-learn.org/stable/supervised_learning.html

# Perceptron implementation is a wrapper around SGDClassifier by fixing
# the loss and learning_rate parameters as:
# SGDClassifier(loss="perceptron", learning_rate="constant")

#  the gradient of the loss is estimated each sample at a time and the model
# is updated along the way with a decreasing strength schedule (aka learning rate).

# loss="perceptron" is the linear loss used by the perceptron algorithm.

# Nota: a volte eta è indicata con la lettera alpha
ppn = Perceptron(eta0=0.1, random_state=1)

# Fit linear model with Stochastic Gradient Descent.
ppn.fit(X_train_std, y_train)

# Predict class labels for samples in X.
y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())


# Accuracy classification score.
# In multilabel classification, this function computes subset accuracy:
# the set of labels predicted for a sample must exactly match the corresponding
# set of labels in y_true.
print('Accuracy (vai accuracy_score): %.3f' % accuracy_score(y_test, y_pred))

# in alternativa impieghiamo score()
# Return the mean accuracy on the given test data and labels.

print('Accuracy (via score): %.3f' % ppn.score(X_test_std, y_test))

# Funzione molto utile per investigare il modello lineare

# Per approfondimenti:
# https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# NEW
plot_decision_regions(X=X_combined_std, y=y_combined, clf=ppn, legend=2)

# OLD
#plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('figures/03_01.png', dpi=300)
plt.show()