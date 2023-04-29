"""
Supervised Learning - Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Access data from sklearn
from sklearn.datasets import load_iris 

iris = load_iris()
iris.feature_names

Data_iris = iris.data

Data_iris = pd.DataFrame(Data_iris,columns = iris.feature_names)

Data_iris["Target"] = iris.target

"""
Exploratory Analysis
"""

plt.scatter(Data_iris.iloc[:,2],Data_iris.iloc[:,3] , c = Data_iris.iloc[:,4])
plt.xlabel("Petal Lenght in cm")
plt.ylabel("Petal Width in cm")
plt.show()

x = Data_iris.iloc[:,0:4]
y = Data_iris.iloc[:,4]

#Modelling

"""
KNN Preamble
"""

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 6, metric = "minkowski", p=1) #default ps is 2 which is euclidean, default for n is 5 if you don't specify it

KNN.fit(x,y)

X_N = np.array([[5.6, 3.4, 1.4, 0.1]])

#Predict
KNN.predict(X_N)

X_N2 = np.array([[7.5, 4, 5.5, 2]])

KNN.predict(X_N2)

X_N3 = np.array([[5.8, 6, 3.2, 5]])

KNN.predict(X_N3)

"""
Training Set and Test Set Creation
"""

 

from sklearn.model_selection import train_test_split 

#Random State can be any number and it's just saying to use the same strategy anytime this particular split is initiated
#stratify=y means 20% of each class of the target should be selected in the test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                    random_state=88, shuffle= True,
                                                    stratify = y)


"""
KNN
"""

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 6, metric = "minkowski", p=1) #default p is 2 which is euclidean and 1 is manhattan distance method, default for n is 5 if you don't specify it

KNN.fit(x_train,y_train)

y_pred = KNN.predict(x_test)


#Classifier model evaluation

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred) #96.7%



"""
Decision Tree
"""

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics

DTC = DecisionTreeClassifier()
DTC.fit(x_train,y_train)

y_pred_dtc= DTC.predict(x_test)

metrics.accuracy_score(y_test, y_pred_dtc)


"""
Cross Validation
"""

#K-Fold Cross Validation: Go through our dataset K times to separate every part of our data into a test data at least once


from sklearn.model_selection import cross_val_score

Scores_DT = cross_val_score(DTC, x,y, cv=10) #CV is cross validation which shows the k value number of folds

Scores_DT.mean() #To get the average of the accuracy scores of the 10 folds above

Scores_KNN = cross_val_score(KNN, x,y, cv=10)

Scores_KNN.mean()

"""
Naive Bayes Classification
"""

from sklearn.naive_bayes import GaussianNB 

NBC = GaussianNB()
NBC.fit(x_train,y_train)

y_pred_NBC = NBC.predict(x_test)

metrics.accuracy_score(y_test,y_pred_NBC)


Scores_NBC = cross_val_score(NBC,x,y,cv=10)

Scores_NBC.mean()


"""
Logistics Regression
"""

from sklearn.datasets import load_breast_cancer

Data_C = load_breast_cancer()

x = Data_C.data
y = Data_C.target


from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, 
                                                    random_state=88, 
                                                    shuffle= True, 
                                                    stratify=y)


from sklearn.linear_model import LogisticRegression

LRC = LogisticRegression()
LRC.fit(x_train,y_train) 

y_pred_LRC = LRC.predict(x_test)

from sklearn.metrics import accuracy_score 

accuracy_score(y_test, y_pred_LRC)

from sklearn.model_selection import cross_val_score

Score_LRC = cross_val_score(LRC,x,y,cv=10)

Score_LRC.mean()



"""
Evaluation Metrics
"""

from sklearn.metrics import confusion_matrix, classification_report

Conf_Mat = confusion_matrix(y_test, y_pred_LRC)

Class_rep = classification_report(y_test, y_pred_LRC)

#Roc Curve

"""
Use Roc Curve & Roc Auc Score in only binary classification
"""

from sklearn.metrics import roc_curve

y_prob = LRC.predict_proba(x_test) #gives me the probability of cancer and not cancer

y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR,TPR)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()



#AUC
from sklearn.metrics import roc_auc_score 

roc_auc_score(y_test,y_prob) #99%

#A great model should have high AUC



"""
Hyper Parameter Optimization - KNN
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Access data from sklearn
from sklearn.datasets import load_iris 

iris = load_iris()
 

x = iris.data

y = iris.target


from sklearn.model_selection import train_test_split 

#Random State can be any number and it's just saying to use the same strategy anytime this particular split is initiated
#stratify=y means 20% of each class of the target should be selected in the test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                    random_state=88, shuffle= True,
                                                    stratify = y)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

KNN_accuracy_test = []
KNN_accuracy_train = []
for i in range(1,50):
    KNNC = KNeighborsClassifier(n_neighbors = i,
                                metric="minkowski", p=2)
    KNNC.fit(x_train,y_train)
    y_pred = KNNC.predict(x_test)
    r_score = recall_score(y_test,y_pred, average="micro")
    a_score = accuracy_score(y_test,y_pred)
    p_score = precision_score(y_test, y_pred, average="micro")
    s_score_test = KNNC.score(x_test,y_test) #Return the mean accuracy on the given test data and labels.
    s_score_train = KNNC.score(x_train,y_train)
#When calculating recall_score for multiclass classification, you must specify the type of average you want e.g. average="micro" as "binary" is the default for binary classification which will throw an error if average isn't define
    KNN_accuracy_test.append(s_score_test)
    KNN_accuracy_train.append(s_score_train)

#np.arange(1,50)
plt.plot(range(1,50),KNN_accuracy_test,label = "test")
plt.plot(range(1,50),KNN_accuracy_train,label = "train")
plt.xlabel("K Number")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()























































































































