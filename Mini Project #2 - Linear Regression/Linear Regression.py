import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import scale
from operator import itemgetter

def linear_regression(X_train, Y_train) :
	# Compute number of observations & features from training data
	num_obs = np.shape(X_train)[0]
	num_features = np.shape(X_train)[1]

	# Create column of 1's equal in size to the number of observations
	ones_col = (np.ones(num_obs).T).reshape((-1,1))

	# Append column of 1's on the left of the features matrix
	X = np.concatenate((ones_col, X_train), axis = 1)

	X_T_X_inv = np.linalg.inv(np.dot((X.T), X))

	# Compute closed form solution for linear regression parameters
	B_hat = np.dot(np.dot((X_T_X_inv), X.T), Y_train)
	
	return B_hat

def linear_regression_predict(X_test, Y_test, B_hat):
	# Compute number of observations & features from training data
	num_obs = np.shape(X_test)[0]
	num_features = np.shape(X_test)[1]

	# Create column of 1's equal in size to the number of observations
	ones_col = (np.ones(num_obs).T).reshape((-1,1))
	X = np.concatenate((ones_col, X_test), axis = 1)
	
	Y_hat = np.dot(X, B_hat)
	avg_test_error = np.mean(np.square(Y_test-Y_hat))
	return avg_test_error

	
def find_zscores(X_train, Y_train, B_hat):

	# Compute number of observations & features from training data
	num_obs = np.shape(X_train)[0]
	num_features = np.shape(X_train)[1]
	
	# Create column of 1's equal in size to the number of observations
	ones_col = (np.ones(num_obs).T).reshape((-1,1))

	# Append column of 1's on the left of the features matrix
	X = np.concatenate((ones_col, X_train), axis = 1)
	
	# Compute fitted values at the training inputs
	Y_train_hat = np.dot(X, B_hat)

	# Estimate standard deviation of Y_train
	sigma_hat = np.sqrt((np.sum(np.square(Y_train-Y_train_hat)))/(num_obs - num_features - 1))
	
	X_T_X_inv = np.linalg.inv(np.dot((X.T), X))

	# Extract diagonal entires
	v = np.diag(X_T_X_inv).reshape(-1,1)

	# Compute standard error for linear regression
	std_error = np.sqrt(v)*sigma_hat

	# Compute z_scores for linear regression
	z_scores = np.divide(B_hat, std_error)
	
	return std_error, z_scores
	

def correlationMatrix(X_train):
	corr_mat = np.corrcoef(X_train, rowvar = 0)
	return corr_mat

	
	
## Main script is below & calls the above functions
	
# Load in feature matrix & labels from dataset
num_features = 7
num_obs = 300
feature_names = ['cylinders', 'displacement', 'horsepower',	'weight', 'acceleration', 'model year',	'origin']

# Load in data and normalize it
mpg_data = scale(np.genfromtxt('auto-mpg.data.csv', delimiter=',', skip_header = 1, dtype = 'float64', usecols = (0,1,2,3,4,5,6,7)))

# Separate training & testing data
X_train = mpg_data[0:num_obs, 0:num_features]
Y_train = mpg_data[0:num_obs, 7].reshape(-1, 1)

X_test = mpg_data[num_obs:, 0:num_features]
Y_test = mpg_data[num_obs:, 7].reshape(-1, 1)

# Compute correlation matrix of features from training data
corr_mat = correlationMatrix(X_train)

# Perform linear regression & obtain beta parameters for further predictions (using all features)
B_hat = linear_regression(X_train, Y_train)

linear_regression_predict(X_test, Y_test, B_hat)

# Find zscores of the features
std_error, zscores = find_zscores(X_train, Y_train, B_hat)
table3_1 = np.concatenate((B_hat, std_error, zscores), axis = 1)

# Perform k-best subset feature selection using an F-value metric
kbest = 2
best_sub_model =  SelectKBest(f_regression, k = kbest)
X_train_kbest = best_sub_model.fit_transform(X_train, np.ravel(Y_train))


# Map features to their scores as determined by F-Values
kbest_scores = {}
for feature_name, score in zip(feature_names, best_sub_model.scores_):
	kbest_scores[feature_name] = score

kbest_Features = sorted(kbest_scores, key = kbest_scores.get, reverse=True)[:kbest]


# Perform linear regression & obtain beta parameters for further predictions (using k-best features)
B_hat_kbest = linear_regression(X_train_kbest, Y_train)

linear_regression_predict(best_sub_model.transform(X_test), Y_test, B_hat_kbest)



# Train the ridge regression model with different regularisation strengths
lambdas = np.logspace(-2, 6, 200)
clf1 = Ridge()
coefs = []
intercepts = []
errors = []

for a in lambdas:
	clf1.set_params(alpha=a)
	clf1.fit(X_train, Y_train)
	coefs.append(clf1.coef_[0])
	intercepts.append(clf1.intercept_[0])
	Y_hat = clf1.predict(X_test)
	avg_test_error = np.mean(np.square(Y_test-Y_hat))
	errors.append(avg_test_error)

# Found by looking at plot
chosen_lambda_pos =  100
lambdas[chosen_lambda_pos]
intercepts[chosen_lambda_pos]
coefs[chosen_lambda_pos]
errors[chosen_lambda_pos]


# Display results
plt.figure(figsize=(20, 6))
plt.subplot(121)
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
ax.legend(feature_names, loc = 3)
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(lambdas, errors)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('error')
plt.title('Mean squared error as a function of the regularization')
plt.axis('tight')
plt.show()


# Train the lasso regression model with different regularisation strengths
lambdas = np.logspace(-4, 1, 200)
clf2 = Lasso()
coefs = []
errors = []
intercepts = []

for a in lambdas:
	clf2.set_params(alpha=a)
	clf2.fit(X_train, Y_train)
	coefs.append(clf2.coef_)
	intercepts.append(clf1.intercept_[0])
	Y_hat = clf2.predict(X_test)
	avg_test_error = np.mean(np.square(Y_test-Y_hat))
	errors.append(avg_test_error)

# Found by looking at plot
chosen_lambda_pos =  80
lambdas[chosen_lambda_pos]
intercepts[chosen_lambda_pos]
coefs[chosen_lambda_pos]
errors[chosen_lambda_pos]

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
ax.legend(feature_names, loc = 3)
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(lambdas, errors)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('error')
plt.title('Mean squared error as a function of the regularization')
plt.axis('tight')
plt.show()