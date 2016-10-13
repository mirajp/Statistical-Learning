import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

with open("train.csv", "r") as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
X_train = np.asarray(data, dtype=float)

with open("trainlabels.csv", "r") as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
Y_train = np.asarray(data, dtype=int)
# 400 of the labels are 1

with open("test.csv", "r") as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
test = np.asarray(data, dtype=float)

Y_train = Y_train.ravel()


#No longer splitting up training set, will compare different classifiers with the scoring function
# Split the training and testing data 60:40 (seed with 42)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.40, random_state=42)
# 238 1's are now in Y_train


"""
clf = svm.SVC()
clf.fit(X_train, Y_train)
#score = clf.score(X_test, Y_test)
#score = 0.992961532822
test_labels = clf.decision_function(test)
"""

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print "Calling grid search"
#5 fold cross validation
clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=6)
print "Starting to fit the data"
clf.fit(X_train, Y_train)
print "Best parameters set found on development set (using precision scoring method, and only 60% of training data):\n"
print clf.best_params_
#Ytest_predictions = clf.predict(X_test)
#print(classification_report(Y_test, Ytest_predictions))