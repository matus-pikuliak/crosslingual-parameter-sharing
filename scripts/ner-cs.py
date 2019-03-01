import codecs, os


def complete(buffer):
    counter = 0
    for word in buffer:
        counter += word.count('<')
        counter -= word.count('>')
        assert(counter >= 0)
    if counter == 0:
        return True
    return False


def get_type(word):
    tags = {
        'P': 'PER',
        'p': 'PER',
        'g': 'LOC',
        'i': 'ORG',
        'ms': 'ORG',
        'oa': 'MISC',
        'op': 'MISC',
    }
    for tag in tags:
        if word.startswith(f'<{tag}'):
            return tags[tag]
    return 'O'


def chunk(word, ite):
    buffer = [word]
    while not complete(buffer):
        buffer.append(next(ite))
    type = get_type(word)

    words = [word for word in buffer if '<' not in word]
    words = [word.replace('>', '') for word in words]
    words = [word for word in words if word]

    return words, type


def line_iterator(line):
    for word in line.split():
        yield word.strip()


def _process_line(line):
    line_ite = line_iterator(line)
    for word in line_ite:
        if word.startswith('<'):
            words, type_ = chunk(word, line_ite)
            if type_ == 'O':
                for word in words:
                    yield word, 'O'
            else:
                yield words[0], f'B-{type_}'
                for word in words[1:]:
                    yield word, f'I-{type_}'
        else:
            yield word, 'O'


def process_line(line):
    return '\n'.join(f'{word} {label}' for word, label in _process_line(line))


raw_file = '/home/fiit/cll-para-sharing/raw_data/ner/cnec2.0/data/plain/named_ent_etest.txt'
result_file = '/home/fiit/test'


with codecs.open(raw_file, encoding='utf-8') as f:
    data = '\n\n'.join(process_line(line) for line in f)

if not os.path.exists(os.path.dirname(result_file)):
    os.makedirs(os.path.dirname(result_file))

with codecs.open(result_file, mode='w', encoding='utf-8') as f:
    f.write(data)



