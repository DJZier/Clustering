# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:58:51 2021

@author: huguet


"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

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
#    2d-4c-no9.arff   xclara.arff
#  2d-4c-no4    spherical_4_3 
# cluto-t8-8k  cluto-t4-8k cluto-t5-8k cluto-t7-8k diamond9 banana
path = './new-data/'
fn = path + "d32.txt"
datanp_init = np.loadtxt(fn, dtype= int)
pca = PCA(n_components=2)
datanp = pca.fit(datanp_init).transform(datanp_init)
knp = list(range(2,20))
print(knp)
coef = []


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

########################################################################
# Calculate average distance between 20 neighbours 

########################################################################

for k in range(2,10) :
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    
    #distances = np.sort(distances, axis=0)
    i=0
    my_mean=[]
    for line in distances :
        my_mean.append(np.mean(line))
    
    my_mean =np.sort(my_mean, axis=0)
    plt.plot(my_mean, label=f"K={k}")

plt.legend()
plt.show()
#Now we have the mean 
# # Run DBSCAN clustering method 
# # for a given number of parameters eps and min_samples
# #
print("With the plotted courb, give a range of ") 


for distance in np.arange(0.02,0.06,0.005):
    #for min_pts in range(2,5) :
        min_pts = 4
        cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data_scaled)
    
        # Plot results
        plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
        title=("Clustering DBSCAN - Epilson="+str(distance)+"Minpt="+str(min_pts))
        plt.title(title)
        plt.show()
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
        n_noise_ = list(cl_pred).count(-1)
        print('The min pts =%d' % min_pts)
        print('The distance =%f' % distance)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
    
    # # Another example
    # distance=0.01
    # min_pts=3
    # cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)
    
    # # Plot results
    # plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
    # plt.title("Clustering DBSCAN - Epilson=0.02 - Minpt=5")
    # plt.show()
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
    # n_noise_ = list(cl_pred).count(-1)
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    
    # # Another example
    # distance=0.02
    # min_pts=5
    # cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)
    
    # # Plot results
    # plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
    # plt.title("Clustering DBSCAN - Epilson=0.02 - Minpt=5")
    # plt.show()
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
    # n_noise_ = list(cl_pred).count(-1)
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

# ########################################################################
# # FIND "interesting" values of epsilon and min_samples 
# # using distances of the k NearestNeighbors for each point of the dataset
# #
# # Note : a point x is considered to belong to its own neighborhood  


