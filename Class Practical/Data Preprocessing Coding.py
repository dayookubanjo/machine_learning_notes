"""
Data Preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data_Set1 = pd.read_csv("Data_Set.csv") 

#Data_test = pd.read_csv("C:/Users/SOK Consulting/Data_Set.csv") 

Data_Set2 = pd.read_csv("Data_Set.csv", header = 2) #header is at index number 2

Data_Set3 = Data_Set2.rename(columns = {"Temperature": "Temp"})

Data_Set4 = Data_Set3.drop("No. Occupants", axis = 1) #Drop a column in a dataframe (axis=1 is column)

#Data_Set3.drop("No. Occupants", axis = 1, inplace = True) #inplace = true means implement my changes to data_set3


#Statistics

#Average , Variance, Standard deviation 


Data_Set5 = Data_Set4.drop(2, axis = 0)

Data_Set6 = Data_Set5.reset_index(drop = True) #Drop the previous index and create a new index


#Print statistical statistics for me to understand my data
Data_Set6.describe()

Min_item = Data_Set6["E_Heat"].min()

#Filter a column 
Data_Set6["E_Heat"][Data_Set6["E_Heat"] == Min_item ]

#Replace values in a column
Data_Set6["E_Heat"].replace(-4,25.666667, inplace = True)

Data_Set6.describe()

#Covariance
#Shows the relation between variables (features). Formula measures the correlation between two variables


Data_Set6.cov() #Actual correlation (How high or low)

#Seaborn to show covariance as an heat map

import seaborn as sn

sn.heatmap(Data_Set6.corr()) #plot correlation


#Missing Values

#Nan = Not a number or blank meaning the cell is filled with something that doesnt make sense
#Null = No value meaning the cell is empty

#Nulls shows as NaN in python


#To convert items that don't make sense to NaN
Data_Set7 = Data_Set6.replace("!", np.NaN)

Data_Set7 = Data_Set7.apply(pd.to_numeric) #Change all my data to numeric

Data_Set7.info() #to view empty values


#Capture Nulls
Data_Set7.isnull()

#Data_Set7.drop([13,22], axis=0, inplace = True)

#Drop rows where null data exists 
Data_Set7.dropna(axis = 0, inplace = True) #delete rows that contain nan values

Data_Set7.reset_index(drop = True, inplace= True)

#Instead of dropping null values, we can replace with the previous value or next value

Data_Set8 = Data_Set7.fillna(method = "ffill") #using the previous observation to fill na

#Data_Set9 = Data_Set7.fillna(method = "bfill") #using the next observation to fill na


#Using a mean, median, most frequent strategy to fill nulls
from sklearn.impute import SimpleImputer 

M_Var = SimpleImputer(missing_values = np.nan, strategy = "mean") #I want to replace nan with the mean of each column

M_Var.fit(Data_Set7)

Data_Set9 = M_Var.transform(Data_Set7) 


"""
Outlier Detection
"""

Data_Set8.boxplot()

Data_Set8.describe()

Data_Set8["E_Plug"].quantile(0.25) #1st Quantile
Data_Set8["E_Plug"].quantile(0.75) #3rd Quantile

#Now check that it has a high potential to be an outlier

#Now that we have confirmed it's an outlier let's handle it

Data_Set8["E_Plug"].replace(120,42, inplace= True) #42 was the value in the previous observation


"""
Concatenating
"""

New_Col = pd.read_csv("Data_New.csv")

Data_Set10 = pd.concat([Data_Set8, New_Col],axis = 1)



"""
Dummy Variables
"""

#Machine doesnt usually understand category data. Switch it to true and false values 1 and 0 that a machine can understand


Data_Set10.info()

Data_Set11 = pd.get_dummies(Data_Set10)

Data_Set11.info()



"""
Normalization
"""
#Normalization is the process of changing variables to the same range so that machines can give them the same level of importance.

from sklearn.preprocessing import minmax_scale, normalize

#First method: Min Max Scale

Data_Set12 = minmax_scale(Data_Set11, feature_range = (0,1)) #Change all the values to a range between 0 and 1


#axis = 0 is when we want to normalize each feature and axis = 1 is when we want to normalize each sample. 
#norm = l1 is Manhattan Distance and l2 is Euclidean distance

Data_Set13 = normalize(Data_Set11, norm = "l2", axis = 0) #l2 is default euclidean

Data_Set13 = pd.DataFrame(Data_Set13, 
                          columns = ["Time","E_Plug","E_Heat","Price","Temp",
                                     "Offpeak", "Peak"])




"""
Other Preprocessing tasks
"""
data.groupby("Married")["LoanAmount"].sum() #Group By
sn.countplot(x=data["Gender"], hue = data["Loan_Status"], data = data) #Bar chart plot of count of hue



























































