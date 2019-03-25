# python WikiExtractor.py -o - --processes 10 --no-templates -q dump.xml > output.txt

import sys

import nltk

tokenizer = nltk.tokenize.TweetTokenizer()

with open(f'{sys.argv[1]}') as f:
    for line in f:
        if line.startswith('<'):
            continue
        if len(line) < 50:
            continue
        sentences = nltk.sent_tokenize(line, sys.argv[2])
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                print(word)
            print()
