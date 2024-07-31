import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits() #loads data
data = scale(digits.data) #your going to scale all features down to make them between -1 and 1. The - .data - part is our features
y = digits.target

k = len(np.unique(y)) #finds the amount of different classifications/classes

k = 10
samples, features = data.shape



def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_), # the y values are our target data. It is compared to the labels that our estimator gave for each of our data.
             metrics.completeness_score(y, estimator.labels_),# because this is an unsupervised learning algorithm, we don't give it y values.
             metrics.v_measure_score(y, estimator.labels_),   # Instead, it generates it's own y value for each testpoint we give it so we don't need split into testing and training data as it doenst know what out test data is.
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean'))) #the absolute distance between two points/vectors in a space


    #trains a bunch of different classifiers and scores them when you call the function

clf = KMeans(n_clusters=k, init="random", n_init=10)

bench_k_means(clf, "1", data) #prints out accuracy scores
