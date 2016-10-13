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


# Do feature reduction using PCA


# Apply the dimensionality reduction to the training set and the test set




tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#5 fold cross validation
#clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=4)
# Swap this out for the previous grid search
clf = svm.SVC()
clf.fit(X_train, Y_train)
#print "Best parameters set found on development set (using precision scoring method):\n"
#print clf.best_params_
#Ytest_predictions = clf.predict(X_test)
#print(classification_report(Y_test, Ytest_predictions))

# Get the metric values for the test data
test_labels = clf.decision_function(test)
test_ids = np.linspace(1, len(test_labels), num=len(test_labels), dtype=int)
test_output = np.column_stack((test_ids,test_labels))

#test_output = np.row_stack((["Id", "Prediction"],test_output))
np.savetxt("testout_withoutpca_withoutgridsearch.csv", test_output, delimiter=",", fmt=("%u", "%.5f"), header="Id, Prediction")
