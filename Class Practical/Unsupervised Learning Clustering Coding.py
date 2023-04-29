"""
Clustering
"""


"""
K-Means Clustering
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


from sklearn.datasets import load_iris

Iris = load_iris()

Data_iris = Iris.data


from sklearn.cluster import KMeans

KMNS = KMeans(n_clusters=3)

KMNS.fit(Data_iris)

Labels = KMNS.predict(Data_iris)

"""
Plotting the clusters
"""

Ctn = KMNS.cluster_centers_ #Find out the centers of the clusters

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.scatter(Ctn[:,2],Ctn[:,3], marker = "o", color="red", s = 120)
plt.xlabel("Petal Length in cm")
plt.ylabel("Petal Width in cm")
plt.show()


"""
Model Evaluation
"""

KMNS.inertia_ #78.85144142614601 - Compare this number to different K values and see where you get the lowest Inertial for best fit

K_inertia = []

for i in range(1,10): #For i in 1 to 9
    KMNS = KMeans(n_clusters=i,random_state = 44)
    KMNS.fit(Data_iris)
    c= KMNS.inertia_
    K_inertia.append(c)
    
K_inertia  

 
plt.plot(range(1,10),K_inertia , marker="o")
plt.xlabel("Number of K")
plt.ylabel("Inertia")
plt.show()



"""
DBSCAN Clustering - To identify outliers/Noise
"""

from sklearn.cluster import DBSCAN
#eps should be big enough but not to big to allow other points in the circle maybe minimum 0.5
DBS = DBSCAN(eps= 0.7, min_samples=4)

DBS.fit(Data_iris)

labels = DBS.labels_ 


plt.scatter(Data_iris[:,2],Data_iris[:,3], c = labels)
plt.show()


"""
Hierrarchichal Clustering
"""

#We will be using scipy here not sklearn

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

HR = linkage(Data_iris,method="complete") #method can be complete, single or average

Dnd = dendrogram(HR)

#to do the clustering to get the labels/targets

labels = fcluster (Z= HR,t=4, criterion="distance") #z is your linkage, t is the maximum intercluster distance that is allowed, criterion is actually the distance
#If you make t too small, you actually don't allow any two samples to form a cluster so you end up with too many clusters
#If t is too much like 10, you may end up with only one cluster because the maximum distance allowed between clusters is too wide
#t=4 is fine
plt.scatter(Data_iris[:,2],Data_iris[:,3], c = labels)
plt.show()



"""
Hyper Parameter Optimization - K-Means
"""


K_inertia = []

for i in range(1,10): #For i in 1 to 9
    KMNS = KMeans(n_clusters=i,random_state = 44)
    KMNS.fit(Data_iris)
    c= KMNS.inertia_
    K_inertia.append(c)
    
K_inertia  

 
plt.plot(range(1,10),K_inertia , marker="o")
plt.xlabel("Number of K")
plt.ylabel("Inertia")
plt.show()




























