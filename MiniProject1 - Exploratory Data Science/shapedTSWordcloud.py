from os import path
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np

d = path.dirname(__file__)

# Read the whole text.
text = open(path.join(d, 'alllyrics.txt')).read()

ts_mask = np.array(Image.open(path.join(d, "ts_mask.2png")))
image_colors = ImageColorGenerator(ts_mask)

# Generate a word cloud image
wordcloud = WordCloud(background_color="white", mask=ts_mask)
wordcloud.generate(text)
wordcloud.to_file(path.join(d, "wordcloud_maskrndcolors.png"))

wordcloud.recolor(color_func=image_colors)
wordcloud.to_file(path.join(d, "wordcloud_masksrccolors.png"))

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
