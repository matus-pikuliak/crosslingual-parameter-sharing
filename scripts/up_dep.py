# Processing UD data into POS file format
import codecs, os


def process(files, result_path):
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    if not isinstance(files, list):
        files = [files]
    sentences = []
    for file_path in files:
        a = 0
        b = 0
        with codecs.open(file_path, encoding='utf-8') as f:
            buffer = []
            for line in f:
                if line.startswith("#") or not line.strip():
                    if len(buffer) > 0:
                        sentences.append("\n".join(buffer))
                        buffer = []
                else:
                    a += 1
                    tokens = line.split('\t')
                    if tokens[0].isdigit():
                        buffer.append("%s\t%s\t%s" % (tokens[1], tokens[6], tokens[7].split(':')[0]))
                    else:
                        b += 1

    with codecs.open(result_path, mode="w", encoding="utf-8") as f:
        data = '\n\n'.join(sentences)
        f.write(data+'\n')

files = [
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-dev.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-test.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-train-c.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-train-l.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-train-m.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Czech-master/cs-ud-train-v.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_English-master/en-ud-dev.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_English-master/en-ud-test.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_English-master/en-ud-train.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_German-master/de-ud-dev.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_German-master/de-ud-test.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_German-master/de-ud-train.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Spanish-master/es-ud-dev.conllu',
# '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Spanish-master/es-ud-test.conllu',
'/media/piko/Data/fiit/data/cll-para-sharing/raw_data/dep_pos/UD_Spanish-master/es-ud-train.conllu',
]

process(files,
        '/media/piko/Data/fiit/data/cll-para-sharing/processed_data/dep/es/train')
