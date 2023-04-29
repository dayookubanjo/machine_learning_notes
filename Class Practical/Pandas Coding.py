"""
Pandas Coding
"""

import pandas as pd

"""
Series
"""

Age = pd.Series([10,20,30,40],index=['age1', 'age2','age3','age4'])
Age.age3
#Filtering a series
Filtered_Age = Age[Age>20]
#Calling values of the series
Age.values
#Calling indices of the series
Age.index
Age.index = ["A1","A2","A3","A4"]


"""
DataFrame
"""

import numpy as np
DF = np.array([[20,10,8],[25,8,10],[27,5,3],[30,9,7]])
Data_Set = pd.DataFrame(DF)
Data_Set = pd.DataFrame(DF,index = ["S1","S2","S3","S4"])
Data_Set = pd.DataFrame(DF,index = ["S1","S2","S3","S4"],columns=["Age","Grade1","Grade2"])

Data_Set["Grade3"] = [9,6,7,10]

# Loc is used to access a group of rows and columns by labels or a boolean array
A =  Data_Set.loc["S2"] #loc is using the index label
Data_Set.iloc[0][3] #iloc is using the row & column index
Data_Set.iloc[0,3]
Data_Set.iloc[:,0]
Filtered_DF=     Data_Set.iloc[:,1:3]

Data_Set.drop("Grade1",axis=1) #drop column
Data_Set.drop("S1",axis=0) #drop row

Data_Set = Data_Set.replace(10,12)
Data_Set = Data_Set.replace({12:10, 9:30})

Data_Set.head(3)
Data_Set.tail(3)
Data_Set.sort_values("Grade1",ascending= False) #Sort column descending order
Data_Set.sort_index(axis = 0, ascending = False) #Sort row or column indexes in ascending or descending order


#Importing csv file
#Set your current location as the working directory


Data = pd.read_csv("Data_Set.csv")
 




