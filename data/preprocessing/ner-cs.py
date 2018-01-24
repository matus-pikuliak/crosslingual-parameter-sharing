import codecs, os

#existuju slova, v ktorych je < a nezacinaju na <
#tieto som nasiel a rucne upravil

def buffer_complete(words):
    sentence = ' '.join(words)
    counter = 0
    for char in sentence:
        if char == '<':
            counter += 1
        if char == '>':
            counter -= 1
        if counter < 0:
            print sentence
            raise 'Error in text formatting'
    if counter == 0:
        return True
    return False

tags = {
    'P': 'PER',
    'p': 'PER',
    'g': 'LOC',
    'i': 'ORG',
    'ms': 'ORG',
    'oa': 'MISC',
    'op': 'MISC',
}


def process_sentence(sentence):
    words = [w.strip() for w in sentence.split(' ')]
    buffer = []
    ner_buffer = []
    in_ner = False
    for word in words:
        if not '<' in word and not in_ner:
            buffer.append((word, 'O'))
        else:
            if not in_ner:
                in_ner = True
                ner_buffer.append(word)
            else:
                ner_buffer.append(word)
                if buffer_complete(ner_buffer):
                    ner_words = (' '.join([w for w in ner_buffer if '<' not in w]).replace('>', '')).split(' ')
                    ner_tags = ['O' for _ in xrange(len(ner_words))]
                    for tag in tags:
                        if ner_buffer[0].startswith('<'+tag):
                            ner_tags = ['I-'+tags[tag] for _ in xrange(len(ner_words))]
                            ner_tags[0] = 'B-'+tags[tag]

                    buffer.extend(zip(ner_words, ner_tags))
                    ner_buffer = []
                    in_ner = False
    if ner_buffer:
        raise 'Something left in NER buffer'
    return buffer


with codecs.open('/media/piko/Data/data/cll-para-sharing/raw_data/ner/cnec2.0/data/plain/named_ent_etest.txt', encoding='utf-8') as f:
    data = '\n\n'.join(['\n'.join([buffer[0]+'\t'+buffer[1] for buffer in process_sentence(line)]) for line in f])

result_path = '/media/piko/Data/data/cll-para-sharing/processed_data/ner/cs/test'

if not os.path.exists(os.path.dirname(result_path)):
    os.makedirs(os.path.dirname(result_path))

with codecs.open(result_path, mode='w', encoding='utf-8') as f:
    f.write(data)



