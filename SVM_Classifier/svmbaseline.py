import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

numPCAComponents = 37

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
# set n_components, chosen to get a sum of variance ratio around 0.95
# this uses auto for svd solver, (most likely random due to the large # of samples and features)
pca = PCA(n_components=numPCAComponents)
pca.fit(X_train)
print "The top ", numPCAComponents, " components yielded the following variance ratios:\n"
print(pca.explained_variance_ratio_)
print "Sum: ", (sum(pca.explained_variance_ratio_))

# Apply the dimensionality reduction to the training set and the test set
X_train = pca.fit_transform(X_train)
test = pca.fit_transform(test)


Cgridspace = np.linspace(10, 100, num=91, dtype=int)
Gammagridspace = np.linspace(0.001, 0.5, num=500, dtype=float)

# Only checking radial kernels
tuned_parameters = [{'kernel': ['rbf'], 'gamma': Gammagridspace,
                     'C': Cgridspace}]

#5 fold cross validation, with 6 jobs/threads
print "Doing grid search with precision as scoring metric\n"
clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=6)
clf.fit(X_train, Y_train)

print "Best parameters set found on development set (using precision scoring method):\n"
print clf.best_params_

# Get the metric values for the test data
print "Evaluating metric scores for the test data"
test_labels = clf.decision_function(test)
test_ids = np.linspace(1, len(test_labels), num=len(test_labels), dtype=int)
test_output = np.column_stack((test_ids,test_labels))

#test_output = np.row_stack((["Id", "Prediction"],test_output))
print "Saving test labels with corresponding ids"
np.savetxt("testout_withpca_withgridsearch.csv", test_output, delimiter=",", fmt=("%u", "%.5f"), header="Id, Prediction")
