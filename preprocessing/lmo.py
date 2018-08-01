import codecs
import sys
import nltk
import time

def write_file(sentences, name):
    with codecs.open(name, mode='w', encoding='utf-8') as f:
        f.write("\n\n".join(["\n".join(sentence) for sentence in sentences]) + "\n")

def process_file(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        sentences = []
        i = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = nltk.word_tokenize(line)
            if float(sum([word.isalpha() for word in words])) / len(words) < 0.5:
                continue
            sentences.append(words)
            i += 1
            if i % 10000 == 0:
                print i
                print time.ctime()
        amount = len(sentences)
        train = sentences[:int(amount*0.90)]
        dev = sentences[int(amount*0.90):int(amount*0.95)]
        test = sentences[int(amount*0.95):]
        write_file(train, 'train')
        write_file(test, 'test')
        write_file(dev, 'dev')
        print len(train)
        print len(dev)
        print len(test)
        print amount



process_file(sys.argv[1])

# rozdel 95 2.5 2.5
# zapis do suboru