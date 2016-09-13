import numpy as np
import nltk
import operator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer

# Statistical Learning
# Mini MATLAB #1: Data Visualization
# Personal Pronoun Analysis of Taylor Swift's Lyrics by Frank Longueira


f_lyrics = open("alllyrics.txt", 'r')
all_lyrics_lines = (f_lyrics.read()).splitlines()
personal_pronouns = {"i":0, "me":0, "we":0, "us":0, "our":0, "my":0, "your":0, "their":0, "they":0, "he":0, "she":0, "them":0, "you":0, "her":0, "him":0,}

# Tokenize file & count pronoun occurrences
for line in all_lyrics_lines:
	tokens = TreebankWordTokenizer().tokenize(line)
	for token in tokens:
		if token.lower() in personal_pronouns:
			personal_pronouns[token.lower()] += 1

personal_pronouns["I"] = personal_pronouns["i"]
del personal_pronouns["i"]

sorted_personal_pronouns = (sorted(personal_pronouns.items(), key=operator.itemgetter(1)))[::-1]

# Print out personal pronoun rankings
for i in range(0, 15):
	print "Personal Pronoun #" + str(i+1) + ":", sorted_personal_pronouns[i]

# Generate a word cloud image based on the rankings
wordcloud = WordCloud(max_font_size=60, relative_scaling=0.5, width = 100, height = 150, scale = 2).generate_from_frequencies(sorted_personal_pronouns)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
