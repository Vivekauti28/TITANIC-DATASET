import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('train.csv')
X=dataset.iloc[:, [2,4,5,6,7,9,11]].values
y=dataset.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer_age= Imputer(missing_values="NaN", strategy="median", axis=0)
imputer_age=imputer_age.fit(X[:, [2]])
X[:, [2]]=imputer_age.transform(X[:, [2]])
dataset['Embarked'].fillna('S', inplace=True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_sex = LabelEncoder()
X[:, 1] = labelencoder_sex.fit_transform(X[:, 1])
labelencoder_embark = LabelEncoder()
X[:, 6] = labelencoder_embark.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)

y_predlogreg=logreg.predict(X_test)

accuracylogreg=accuracy_score(y_true=y_test,y_pred=y_predlogreg)*100
                       
from sklearn.ensemble import RandomForestClassifier
model_random = RandomForestClassifier(n_estimators = 700,
                                     oob_score=True,
                                     random_state=0,
                                      min_samples_split=10                                  
                                )
model_random.fit(X_train,y_train)
y_predrandom=model_random.predict(X_test)
accuracyrandom=accuracy_score(y_true=y_test,y_pred=y_predrandom)*100

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predsvc=svc_model.predict(X_test)
accuracysvc=accuracy_score(y_true=y_test,y_pred=y_predsvc)*100
                       
from sklearn.neighbors import KNeighborsClassifier
knn_model  = KNeighborsClassifier(n_neighbors = 14)
knn_model.fit(X_train, y_train)
y_predknn=knn_model.predict(X_test)
accuracyknn=accuracy_score(y_true=y_test,y_pred=y_predknn)*100

from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)
y_prednaive=model_naive.predict(X_test)
accuracynaive=accuracy_score(y_true=y_test,y_pred=y_prednaive)*100
  
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini',
                                    min_samples_split=12,
                              max_features='auto',
                              min_samples_leaf=12)
tree_model.fit(X_train, y_train)
y_predtree=tree_model.predict(X_test)
accuracytree=accuracy_score(y_true=y_test,y_pred=y_predtree)*100
                       
from sklearn.ensemble import AdaBoostClassifier
model_ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1)
model_ada.fit(X_train, y_train)
y_pred_ada=model_ada.predict(X_test)
accuracy_ada=accuracy_score(y_true=y_test,y_pred=y_pred_ada)*100

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model_linear_d= LinearDiscriminantAnalysis()
model_linear_d.fit(X_train,y_train)
y_pred_lda=model_linear_d.predict(X_test)
accuracy=accuracy_score(y_true=y_test,y_pred=y_pred_lda)*100