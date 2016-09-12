#!/usr/bin/env python
from nltk.corpus import brown
from collections import Counter
from itertools import *
from string import lower
import operator
from math import log

def print_list(ls):
    for word_count, i in zip(ls, range(1, 21)):
        
        print i, ". ", word_count[0], word_count[1]

with open("alllyrics.txt") as f:
    swift_words = map(lower, f.read().split())

    
swift_word_counts = Counter(swift_words)
#print swift_word_counts

brown_word_counts = Counter(map(lower, brown.words()))
#print(brown_word_counts)


swifty_scores = {}

for word, count in swift_word_counts.iteritems():
    if (count >= 3):
        brown_count = brown_word_counts[word]
        if (brown_count > 0):
            swifty_scores[word] = log(count / float(brown_count))


sorted_scores = sorted(swifty_scores.items(), key=operator.itemgetter(1))

print """

A word's "Swifty" score is defined as log(N_swift / N_brown), where
N_swift is the number of occurrences in the Swift lyric set and
N_brown is the number of occurrences in the Brown corpus.

"""


print "Top 20 Swiftiest words: "
top_20 = sorted_scores[-20:]
top_20.reverse()
print_list(top_20)
print ""
print "Top 20 least-Swifty words: "
print_list(sorted_scores[0:20])

