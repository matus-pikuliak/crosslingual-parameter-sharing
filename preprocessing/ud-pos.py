# Processing UD data into POS file format
# Example:
#
# word1 tag1
# word2 tag2
#
# word1 tag3
# word2 tag4
import codecs, os


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
                    if len(buffer) > 0:
                        sentences.append("\n".join(buffer))
                        buffer = []
                else:
                    tokens = line.split('\t')
                    if tokens[0].isdigit():
                        buffer.append(tokens[1]+"\t"+tokens[3])
                    else:
                        print(tokens[0])
            if len(buffer) > 0:
                sentences.append(list(buffer))

    with codecs.open(result_path, mode="w", encoding="utf-8") as f:
        data = '\n\n'.join(sentences)
        f.write(data+'\n')


process('/media/piko/Data/data/cll-para-sharing/raw_data/dep_pos/UD_English-master/en-ud-dev.conllu',
        '/media/piko/Data/data/cll-para-sharing/processed_data/pos/en/dev')
