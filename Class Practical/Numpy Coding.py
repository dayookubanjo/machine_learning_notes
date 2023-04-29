"""
Numpy Coding
"""

import numpy as np


NumP_Array = np.array([[1,2,3],[4,5,6]])

NP1 = np.array([[1,3],[4,5]])
NP2 = np.array([[3,4],[5,7]])
#Matrix multiplication: Row 1, Column 1 = 1*3 + 3*5 = 18
MNP = NP1@NP2
#use the dot attribute of numpy to multiply NP1 and NP2 in a matrix way sames as MNP
MNP3 = np.dot(NP1,NP2) 
MNP2 = NP1*NP2
#use the multiply attribute of numpy to multiply NP1 and NP2 in a normal way sames as MNP2
MNP4 = np.multiply(NP1,NP2)

Sum1 = NP1 + NP2
Sub1 = NP1 - NP2
Sub2 = np.subtract(NP1,NP2)
El_Sum = np.sum(NP1) #add all the elements of an array together

#Broadcasting
Broad_Nump = NP1+3
NP3 = np.array([[3,4]])
NP1+NP3

#Divide

D= np.divide([12,14,16],5)
D1= np.floor_divide([12,14,16],5)
D2 = np.divide(NP1,3)

np.math.sqrt(10) #Numpy Math Class
 
#Generate distribution
ND = np.random.standard_normal((3,4))  #Generate random numbers in a form of normal distribution array
UD = np.random.uniform(1,12,(3,4))  #Generate random numbers in a form of uniform distribution array
RandFloat_Array = np.random.rand(3,5) #Generate float numbers
RandInt_Array = np.random.randint(1,50,(2,5)) #Generates integer numbers
Ze = np.zeros((3,4)) #Generates an array of zeros
Ones = np.ones((3,4)) #Generates an array of ones


Filter_Ar = np.logical_and(RandInt_Array>30, RandInt_Array<50) #Returns an array with true or false depending on where filter applied is true
F_RandInt_Array = RandInt_Array[Filter_Ar]


#Some statistics

Data_N = np.array([1,3,4,5,7,9])
Mean_N = np.mean(Data_N)
Median_N = np.median(Data_N)
Var_N = np.var(Data_N)
#Std
SD_N = np.std(Data_N)
SD_N1 = np.math.sqrt(Var_N)

#Multidimensional
NumP_Array = np.array([[1,2,3],[4,6,7]])
Var_Nump = np.var(NumP_Array, axis=1) #axis is the direction you want 1 is by rows, 0 is by columns
Var_Nump2 = np.var(NumP_Array, axis=0) #axis is the direction you want 1 is by rows, 0 is by columns
