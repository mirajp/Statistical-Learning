from html.parser import HTMLParser
import requests
import urllib.request
class MyHTMLParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
                # Only parse the 'anchor' tag.
                if tag == "a":
                        # Check the list of defined attributes.
                        for name, value in attrs:
                                # If href is defined, print it.
                                if name == "href" and (value.find("lyrics/taylorswift")) != -1:
                                        print(value)
                                        exit
                                        lyricsurl = value.replace("..", "http://www.azlyrics.com")
                                        songtitle = lyricsurl.replace("http://www.azlyrics.com/lyrics/taylorswift/", "")
                                        songtitle = songtitle.replace(".html", "")
                                        print(songtitle + ": " + lyricsurl)
                                        fp =  urllib.request.urlopen(lyricsurl)
                                        mybytes = fp.read()
                                        fp.close()
                                        lyricshtml = mybytes.decode("utf8")
                                        lyrics = lyricshtml[lyricshtml.find("Sorry about that. -->")+22:lyricshtml.find("<!-- MxM banner -->")]
                                        f = open(songtitle + '.txt', 'w')
                                        f.write(lyrics)
                                        f.close()
                                        break
                                        
url ='http://www.azlyrics.com/t/taylorswift.html'
fp =  urllib.request.urlopen(url)
mybytes = fp.read()
mystr = mybytes.decode('utf8')
fp.close()
parser = MyHTMLParser()
parser.feed(mystr)
