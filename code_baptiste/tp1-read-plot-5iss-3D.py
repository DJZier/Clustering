# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff

path = './artificial/'
databrut = arff.loadarff(open(path+"elliptical_10_2.arff", 'r'))
datanp = np.array([[x[0],x[1],x[2]] for x in databrut[0]])
#print(databrut)
#print(datanp)

##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")

axe=plt.axes(projection='3d')
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne]
f2 = datanp[:,2]
#print(f0)
#print(f1)
axe.scatter3D(f0, f1, f2, c=f0, s=8)
axe.title("Donnees initiales")
axe.show()

########################################################################
# AUTRES VISUALISATION DU JEU DE DONNEES
# (histogrammes par exemple,)
# But : essayer d'autres types de plot 
########################################################################

########################################################################
# STANDARDISER ET VISUALISER 
# But : comparer des méthodes de standardisation, ...
########################################################################
