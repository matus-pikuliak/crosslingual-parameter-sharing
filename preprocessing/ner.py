# Processing various file format into uniform NER file format

import codecs, os

tags = [
    'B-LOC',
    'B-ORG',
    'B-OTH',
    'B-PER',
    'I-LOC',
    'I-ORG',
    'I-OTH',
    'I-PER',
    'B-MISC',
    'I-MISC',
    'O'
]


def process(files, result_path):
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    if not isinstance(files, list):
        files = [files]
    sentences = []
    for file_path in files:
        with codecs.open(file_path, encoding='utf-8') as f:
            buffer = []
            for line in f:
                if line.startswith("#") or not line.strip():
                    if len(buffer) > 2:
                        i = 0
                        for word in buffer:
                            if any(char.isdigit() for char in word.split('\t')[0]):
                                i += 1
                        if i < float(len(buffer))/3:
                            sentences.append("\n".join(buffer))
                    buffer = []
                else:
                    tokens = line.split(' ')
                    tag = tokens[3].strip()
                    if tag in tags:
                        if tag == 'B-OTH':
                            tag = 'B-MISC'
                        if tag == 'I-OTH':
                            tag = 'I-MISC'
                        buffer.append(tokens[0]+"\t"+tag)
                    else:
                        print(tag)
                    #if not tokens[0].isdigit():
                    #    print tokens[0]
            if len(buffer) > 2:
                i = 0
                for word in buffer:
                    if any(char.isdigit() for char in word.split('\t')[0]):
                        i += 1
                if i < float(len(buffer)) / 3:
                    sentences.append("\n".join(buffer))


            with codecs.open(result_path, mode="w", encoding="utf-8") as f:
                data = '\n\n'.join(sentences)
                f.write(data+'\n')

process('/media/piko/Data/data/cll-para-sharing/raw_data/ner/en/eng.testb',
        '/media/piko/Data/data/cll-para-sharing/processed_data/ner/en/test')
