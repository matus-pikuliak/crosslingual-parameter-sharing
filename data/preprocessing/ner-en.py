import codecs
import csv

filename = '/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/raw_data/ner/en/utf8.csv'

tags = set()

code = {
    'B-eve': 'B-MISC',
    'B-nat': 'B-MISC',
    'B-org': 'B-ORG',
    'B-gpe': 'B-MISC',
    'B-tim': 'O',
    'B-art': 'B-MISC',
    'B-per': 'B-PER',
    'B-geo': 'B-LOC',
    'I-eve': 'I-MISC',
    'I-nat': 'I-MISC',
    'I-org': 'I-ORG',
    'I-gpe': 'I-MISC',
    'I-tim': 'O',
    'I-art': 'I-MISC',
    'I-per': 'I-PER',
    'I-geo': 'I-LOC',
    'O': 'O',
}

sentences = []
with codecs.open(filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    reader.next()
    buffer = []
    for line in reader:
        tags.add(code[line[3]])
        if line[0].startswith('S') and buffer:
            sentences.append(''.join(buffer))
            buffer = []
        buffer.append(line[1]+'\t'+code[line[3]]+'\n')
sentences.append(''.join(buffer))
pocet = len(sentences)
first = int(pocet*0.8)
second = int(pocet*0.9)
a = sentences[:first]
b = sentences[first:second]
c = sentences[second:]

with codecs.open('/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/raw_data/ner/en/train', 'w', encoding='utf-8') as f:
    f.write('\n'.join(a))
with codecs.open('/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/raw_data/ner/en/dev', 'w', encoding='utf-8') as f:
    f.write('\n'.join(b))
with codecs.open('/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/raw_data/ner/en/test', 'w', encoding='utf-8') as f:
    f.write('\n'.join(c))

#print '\n'.join(sentences)

# eve - MISC - event
# nat - MISC - prirodne javy, napr. hurikany, choroby
# org - ORG - organizecie
# gpe - MISC - nacionality
# tim - O - casy (wednesday, ...)
# art - MISC - artefakty?
# per - PER - persons
# geo - LOC - geograficke veci
