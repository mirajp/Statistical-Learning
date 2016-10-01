#!/usr/bin/env python

NEGATIVE_CLASS = 'B'
POSITIVE_CLASS = 'M'



import sys
from sklearn.linear_model import SGDClassifier 
from sklearn.preprocessing import StandardScaler


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
    predictions = clf.predict(scaled_test_features)
    print_accuracy(test_classes, predictions)

    
    
def print_accuracy(ground_truth, predictions):
    true_pos, true_neg, false_pos, false_neg = 0,0,0,0

    for t, p in zip(ground_truth, predictions):
        true_pos += (t == 1 and p == 1)  
        false_pos += (t == 1 and p == 0)
        true_neg += (t == 0 and p == 0)
        false_neg += (t == 0 and p == 1)
            
    print "Correctly classified malginant:", true_pos
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
