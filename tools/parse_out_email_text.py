#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        and return a string that contains all the words
        in the email (space-separated) 

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        """

    f.seek(0)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(
            string.maketrans("", ""), string.punctuation)

        words = ""
        # split the text string into individual words, stem each word,
        # and append the stemmed word to words (make sure there's a single
        # space between each stemmed word)
        stemmer = SnowballStemmer("english")

        ar = text_string.split()
        br = []
        for i in range(len(ar)):
            br.append(stemmer.stem(ar[i]))
        for i in range(len(br)):
            words += br[i]
            words += " "

    return words


def main():
    ff = open("test_email.txt", "r")
    text = parseOutText(ff)
    print text


if __name__ == '__main__':
    main()
