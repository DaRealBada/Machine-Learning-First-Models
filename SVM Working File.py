# SVMs are used when there is no clear linear pattern in the data which is why SVMs would produce higher accuracy than KNN.
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer() #library from sklearn which has loads of data already
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant','benign']

#clf, short for classifier, is used to store trained model values to be used for further prediction. It is an estimator instance
#other kernels can be used like poly, sigmoid,precomputed..... - C=2, refers to the soft margin where 0 is a hard margin and 2 is softer
#for kernels which take more time we can add a degree to alter the processing time - svm.SVC(kernel = "poly", degree=2)

#clf = KNeighborsClassifier(n_neighbors=8) #KNN could be used rather than SVM
clf = svm.SVC(kernel = "linear", C = 2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)