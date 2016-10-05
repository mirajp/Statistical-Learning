#!/usr/bin/env python2.7
import sys
import collections
import math
from sklearn.linear_model import SGDClassifier 
from sklearn.preprocessing import StandardScaler
import numpy as np

NEGATIVE_CLASS = 'B' # Benign
POSITIVE_CLASS = 'M' # Malignant

# These parameters seemed to work well...
eta0 = 0.0001
max_iterations = 1800
stopping_val = 1e-12
# Only set one (or 0) of these to true at a time...
L1_PENALTY = False
L2_PENALTY = False

def main():
    if len(sys.argv) != 3:
        print "Usage: " + sys.argv[0] + " <training file> <testing file>"
        exit(1)

    ### Parse and normalize the training data ###
    
    features, classes = parse_data(open(sys.argv[1], 'r'))
    scaler = StandardScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    
    ### Train the baseline classifier ###
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(scaled_features, classes)

    ### Parse and normalize the test data ###
    test_features, test_classes = parse_data(open(sys.argv[2], 'r'))
    scaled_test_features = scaler.transform(test_features)
    
    ### classify and print stats for the baseline classifier ###
    print "SKLEARN BUILT-IN CLASSIFIER RESULTS"
    print "Settings: ", clf
    predictions = clf.predict(scaled_test_features)
    print_accuracy(test_classes, predictions)

    ### Train my classifier ###
    scaled_features_plus_intercept = np.ones((scaled_features.shape[0], scaled_features.shape[1] + 1))
    scaled_features_plus_intercept[:, 1:] = scaled_features
    weights, t = train(scaled_features_plus_intercept, classes)
    
    ### Test my classifier ###
    print "\n\nMY CLASSIFIER RESULTS"
    print "Trained for", t, "iterations."
    print "Learning rate:", eta0
    scaled_test_features_plus_intercept = np.ones((scaled_test_features.shape[0], scaled_test_features.shape[1] + 1))
    scaled_test_features_plus_intercept[:, 1:] = scaled_test_features

    my_predictions = [classify(weights, instance) for instance in scaled_test_features_plus_intercept]
    print_accuracy(test_classes, my_predictions)

def train(X, classes):
    # randomly initialize weights between 0 and 1.
    weights = np.random.rand((X.shape[1]), 1)
    diff = stopping_val + 1.

    
    t = 0
    rows = range(len(X))
    while (True):

        np.random.shuffle(rows)
        X = X[rows, :]
        new_classes = [None] * len(classes)
        for i in xrange(len(classes)):
            new_classes[i] = classes[rows[i]]
        classes = new_classes

        for i in xrange(len(X)):
            prob = 1 / (1 + np.exp(-X[i,:].dot(weights)))
            error = classes[i] - prob
            error_product = (X[i, :] * error).reshape(X.shape[1],1)
            diff = sum(map(np.abs, error_product))
            if ((diff < stopping_val) or (t > max_iterations)) and t > 3 * len(X):
                break

            weights[0] += eta0 * error_product[0]

            t += 1
            if L1_PENALTY:
                for i in xrange(1, len(weights)):
                    weights[i] += eta0 * error_product[i] - np.sign(weights[i]) * eta0
            elif L2_PENALTY:
                for i in xrange(1, len(weights)):
                    weights[i] += eta0 * error_product[i] - (weights[i] * eta0)
            else:
                for i in xrange(1, len(weights)):
                    weights[i] += eta0 * error_product[0]
        if ((diff < stopping_val) or (t > max_iterations)) and  t >  3 * len(X):
            break

    return weights, t



# calculate the probability and classify as malignant if p > 0.5
def classify(W, X):
    return 1 if 1 / (1 + np.exp(-X.dot(W))) > 0.5 else 0

def print_accuracy(ground_truth, predictions):
    true_pos, true_neg, false_pos, false_neg = 0,0,0,0

    for t, p in zip(ground_truth, predictions):
        true_pos += (t == 1 and p == 1)  
        false_pos += (t == 1 and p == 0)
        true_neg += (t == 0 and p == 0)
        false_neg += (t == 0 and p == 1)
            
    print "Correctly classified malignant:", true_pos
    print "Incorrectly classified malignant:", false_pos
    print "Correctly classified benign:", true_neg
    print "Incorrectly classified benign:", false_neg
    print "TOTAL CORRECT:",  ( true_pos + true_neg ), "/", len(predictions), "=", float(true_pos + true_neg) / float(len(predictions)) 
    
    
def parse_data(input_stream):
    classes = []
    features = []
    for line in input_stream:
        fields = line.strip('\n').split(',')
        if fields[1] == NEGATIVE_CLASS:
            classes.append(0)
        else:
            classes.append(1)
        features.append(map(float, fields[2:]))

    return features, classes


if __name__ == '__main__':
    main()
