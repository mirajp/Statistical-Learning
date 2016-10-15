import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from random import randint

mode = 1
#mode = 2
#mode = 3

logFileName = "Mode" + str(mode) + "_pcagridresults.txt"
predictionsFileName = "Mode" + str(mode) + "_pcagrid_predictions.csv"
logFile = open(logFileName, 'w')

Cgridspace = np.linspace(10, 100, num=91, dtype=int)
Gammagridspace = np.linspace(0.001, 0.5, num=500, dtype=float)

# Only checking radial kernels
tuned_parameters = [{'kernel': ['rbf'], 'gamma': Gammagridspace,
                     'C': Cgridspace}]

# The number of principal components to account for 95% of the variance will
# probably depend on the size of the total training set, but 28 seems good enough for now
numPCAComponents = 28

# Load the training and test data
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

with open("test.csv", "r") as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
test = np.asarray(data, dtype=float)

# If its not ravelled, it is automatically done so with a warning
Y_train = Y_train.ravel()

numRareClass = sum(Y_train)

X_train_common = []
Y_train_common = []
X_train_rare = []
Y_train_rare = []

# Bin the training by class
for obsIter in range(len(Y_train)):
	if (Y_train[obsIter] == 1):
		X_train_rare.append(X_train[obsIter])
		Y_train_rare.append(Y_train[obsIter])
	else:
		X_train_common.append(X_train[obsIter])
		Y_train_common.append(Y_train[obsIter])

newX_train = X_train_rare
newY_train = Y_train_rare

# Add proportionate amount of common class observations
if (mode == 2):
	numCommonToAdd = 2*numRareClass # 66-33
	logFile.write("Adding 2x number of common class obs as rare\n")
elif (mode == 3):
	numCommonToAdd = 3*numRareClass # 75-25
	logFile.write("Adding 3x number of common class obs as rare\n")
else:
	numCommonToAdd = numRareClass # 50-50
	logFile.write("Adding same number of common class obs as rare\n")

for obsIter in range(numRareClass):
	randIndex = randint(0, len(X_train_common))
	newX_train.append(X_train_common[randIndex])
	newY_train.append(Y_train_common[randIndex])
	X_train_common.pop(randIndex)
	Y_train_common.pop(randIndex)

pca = PCA(n_components=numPCAComponents)
pca.fit(newX_train)
print "The top ", numPCAComponents, " components yielded the following variance ratios:\n"
logFile.write("The top ", numPCAComponents, " components yielded the following variance ratios:\n")

print pca.explained_variance_ratio_
logFile.write(pca.explained_variance_ratio_)
logFile.write("\n")

print "Sum: ", (sum(pca.explained_variance_ratio_))
logFile.write("Sum: ", (sum(pca.explained_variance_ratio_)))
logFile.write("\n")

# Apply the dimensionality reduction to the training set and the test set
newX_train = pca.fit_transform(newX_train)
test = pca.fit_transform(test)

#5 fold cross validation, with 6 jobs/threads
print "Doing grid search with precision as scoring metric\n"
logFile.write("Doing grid search with precision as scoring metric\n")

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=6)
clf.fit(newX_train, newY_train)

print "Best parameters set found on development set (using precision scoring method):\n"
logFile.write("Best parameters set found on development set (using precision scoring method):\n")
print clf.best_params_
logFile.write(clf.best_params_)
logFile.write("\n")

# Get the metric values for the test data
print "Evaluating metric scores for the test data"
logFile.write("Evaluating metric scores for the test data\n")

test_labels = clf.decision_function(test)
test_ids = np.linspace(1, len(test_labels), num=len(test_labels), dtype=int)
test_output = np.column_stack((test_ids,test_labels))

#test_output = np.row_stack((["Id", "Prediction"],test_output))
print "Saving test labels with corresponding ids"
logFile.write("Saving test labels with corresponding ids\n")
logFile.close()

np.savetxt(predictionsFileName, test_output, delimiter=",", fmt=("%u", "%.5f"), header="Id, Prediction")
