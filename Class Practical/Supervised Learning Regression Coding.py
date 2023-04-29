"""
Supervised Learning - Regression
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.datasets import load_boston

Boston = load_boston()

x = Boston.data 
y = Boston.target 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=88)
              
                                                    
"""
Data Preprocessing - Normalization (x_train, x_test, y_train)
""")

#During Regression, you should normalize the data 

#This is not a must for multiple linear regression because we have different constants for each feature (a1...an)



from sklearn.preprocessing import MinMaxScaler 

Sc = MinMaxScaler(feature_range = (0,1))
x_train = Sc.fit_transform(x_train )
x_test = Sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1) #reshape y_train first before normalizing to eliminate error
y_train = Sc.fit_transform(y_train)


"""
Linear regression modelling
"""

from sklearn.linear_model import LinearRegression 

LR = LinearRegression()
LR.fit(x_train,y_train)


y_pred = LR.predict(x_test)

y_pred = Sc.inverse_transform(y_pred)


"""
Evaluation Metrics: Evaluate the goodness of our linear model
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import math 



MAE = mean_absolute_error(y_test,y_pred) #4.21 

MSE = mean_squared_error(y_test,y_pred) #32.42 

RMSE = math.sqrt(MSE) #5.62

R2 = r2_score(y_test, y_pred) #57.6%

Ytest_Mean = y_test.mean()  #22.85  
Ytest_Std = y_test.std() #8.74

#MAE & RMSE 


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean( np.abs((y_true - y_pred)/y_true)) * 100


MAPE = mean_absolute_percentage_error(y_test,y_pred) #48%

Intercept = LR.intercept_
Coefficient = LR.coef_ 


"""
Polynomial Linear Regression : Curved line instead of straight line
"""

 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.datasets import load_boston

Boston = load_boston()

Data_Boston = Boston.data

x = Data_Boston[:,5]
y = Boston.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 88)


"""
Data Preprocessing
"""

from sklearn.preprocessing import PolynomialFeatures

Poly_P = PolynomialFeatures(degree=2)
#We want to make the x_train one feature instead of (379,) i.e. to (379,1)
x_train = x_train.reshape(-1,1)

#Transform x_train to polynomial format with 3 dimensions (379,3) from (379,1)
Poly_X = Poly_P.fit_transform(x_train)


from sklearn.linear_model import LinearRegression 

LR = LinearRegression()

Poly_L_R = LR.fit(Poly_X, y_train)


#Before testing your model, change x_test to polynomial format
x_test = x_test.reshape(-1,1)
Poly_X_T = Poly_P.fit_transform(x_test)


y_pred = Poly_L_R.predict(Poly_X_T) 





"""
Evaluation Metrics: Evaluate the goodness of our linear model
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import math 



MAE = mean_absolute_error(y_test,y_pred) #4.21 3.84

MSE = mean_squared_error(y_test,y_pred) #32.42 32.65

RMSE = math.sqrt(MSE) #5.62 5.71

R2 = r2_score(y_test, y_pred) #57.6% 57.26%

Ytest_Mean = y_test.mean()  #22.85  
Ytest_Std = y_test.std() #8.74

#MAE & RMSE 


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean( np.abs((y_true - y_pred)/y_true)) * 100


MAPE = mean_absolute_percentage_error(y_test,y_pred) #48% 19.55%

Intercept = Poly_L_R.intercept_ #56.84
Coefficient = Poly_L_R.coef_ 






"""
Random Forest Regression
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.datasets import load_boston

Boston = load_boston()

x = Boston.data 
y = Boston.target 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=88)
              
                                                    
"""
Data Preprocessing - Normalization (x_train, x_test, y_train)
""")

#During Regression, you should normalize the data 

#This is not a must for multiple linear regression because we have different constants for each feature (a1...an)



from sklearn.preprocessing import MinMaxScaler 

Sc = MinMaxScaler(feature_range = (0,1))
x_train = Sc.fit_transform(x_train )
x_test = Sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1) #reshape y_train first before normalizing to eliminate error
y_train = Sc.fit_transform(y_train)


from sklearn.ensemble import RandomForestRegressor

#n_estimators=100,max_depth = 15

RFR = RandomForestRegressor(n_estimators=500,max_depth = 10, random_state=33)

RFR.fit(x_train,y_train)

y_pred = RFR.predict(x_test)

y_pred = y_pred.reshape(-1,1)

y_pred = Sc.inverse_transform(y_pred)




"""
Evaluation Metrics 
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import math 



MAE = mean_absolute_error(y_test,y_pred) #4.21 3.84 2.62 2.78 2.62

MSE = mean_squared_error(y_test,y_pred) #32.42 32.65 12.73 15.37 12.16

RMSE = math.sqrt(MSE) #5.62 5.71 3.57 3.92 3.49

R2 = r2_score(y_test, y_pred) #57.6% 57.26% 83% 79.87% 84%

Ytest_Mean = y_test.mean()  #22.85  
Ytest_Std = y_test.std() #8.74

#MAE & RMSE 


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean( np.abs((y_true - y_pred)/y_true)) * 100


MAPE = mean_absolute_percentage_error(y_test,y_pred) #48% 19.55% 44.56% 45.42% 44.44.63%




"""
Support Vector Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.datasets import load_boston

Boston = load_boston()

x = Boston.data 
y = Boston.target 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=88)
              
                                                    
"""
Data Preprocessing - Normalization (x_train, x_test, y_train)
""")

#During Regression, you should normalize the data 

#This is not a must for multiple linear regression because we have different constants for each feature (a1...an)



from sklearn.preprocessing import MinMaxScaler 

Sc = MinMaxScaler(feature_range = (0,1))
x_train = Sc.fit_transform(x_train )
x_test = Sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1) #reshape y_train first before normalizing to eliminate error
y_train = Sc.fit_transform(y_train)

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel="rbf")
Regressor_SVR.fit(x_train,y_train)

y_pred = Regressor_SVR.predict(x_test)

y_pred = y_pred.reshape(-1,1)
y_pred = Sc.inverse_transform(y_pred)



"""
Evaluation Metrics 
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import math 



MAE = mean_absolute_error(y_test,y_pred) #4.21 3.84 2.62 2.78 2.62 SVM=2.62

MSE = mean_squared_error(y_test,y_pred) #32.42 32.65 12.73 15.37 12.16 SVM=12.49

RMSE = math.sqrt(MSE) #5.62 5.71 3.57 3.92 3.49 SVM= 3.53

R2 = r2_score(y_test, y_pred) #57.6% 57.26% 83% 79.87% 84% SVM=84%

Ytest_Mean = y_test.mean()  #22.85  
Ytest_Std = y_test.std() #8.74

#MAE & RMSE 


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean( np.abs((y_true - y_pred)/y_true)) * 100


MAPE = mean_absolute_percentage_error(y_test,y_pred) #48% 19.55% 44.56% 45.42% 44.44.63% SVM = 45%




"""
Hyper Parameter Optimization
"""

"""
Hyper Parameter Optimization - SVR
"""


#Former Approach
from sklearn.svm import SVR

Regressor_SVR = SVR(kernel="rbf")
Regressor_SVR.fit(x_train,y_train)

y_pred = Regressor_SVR.predict(x_test)

y_pred = y_pred.reshape(-1,1)
y_pred = Sc.inverse_transform(y_pred)



#New Approach to use hyper parameter optimization
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

#define the parameters we want to optimize in a dictionary
parameters = {"kernel":["rbf","linear"],
              "gamma":["scale","auto"]} #1,0.1,0.01

grid = GridSearchCV(SVR(), param_grid=parameters, refit=True,
                    verbose = 2, scoring= "neg_mean_squared_error")

grid.fit(x,y)

best_params = grid.best_params_


#refit = True goes through the whole dataset to find the best parameters for you.
#verbose refers to how you want to see the calculation in your console


















































