from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data") #cvs can be used for any type of data
# print(data.head())
le = preprocessing.LabelEncoder() #converts string data into numeric data. from KNN classifier

#The method fit_transform() takes a list (each of our columns)
# and will return to us an array containing our new values.
buying = le.fit_transform(list(data["buying"])) #turns buying data from car data into a list and then transforms it into numeric values.
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  #optional

X = list(zip(buying, maint, door, persons, lug_boot, safety)) #to recombine our data into a feature list and a label list
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9) #chooses model
model.fit(x_train, y_train) #trains model
acc = model.score(x_test,y_test) #tests accuracy of model

predicted = model.predict(x_test)

names = ["unacc","acc","good","v_good"]

for x in range(len(x_test)):
    print("Predicted data:", names[predicted[x]],"Data:", x_test[x], "Actual:", names[y_test[x]]) #names is optional and is wrapped around the variables
    n = model.kneighbors([x_test[x]], 9, True)                        #but it is optional, and numeric scores would be shown rather than one of the names
    print("N:", n)