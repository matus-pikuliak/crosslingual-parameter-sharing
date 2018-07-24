import codecs
import nltk

nli_fold = '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/nli/snli_1.0/snli_1.0/snli_1.0_%s.txt'
role = 'train'

with codecs.open(nli_fold % role, encoding='utf-8') as f:
    txts = []
    f.next()
    for line in f:
        o = line.split('\t')
        tag, premise, hypothesis = o[0], o[5], o[6] # 1,2 pre ine jazyky ako
        if tag == u'-':
            continue
        premise = nltk.word_tokenize(premise)
        hypothesis = nltk.word_tokenize(hypothesis)
        hypothesis[0] = "%s\t%s" % (hypothesis[0], tag)
        lines = premise + hypothesis
        txt = '\n'.join(lines)
        txts.append(txt)

target_dir = '/media/piko/Data/fiit/data/cll-para-sharing/processed_data/nli'

with codecs.open("%s/%s/%s" % (target_dir, 'en', role), 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(txts))



exit()

# TEST DATA NLI

lng = 'en'
nli_file = '/media/piko/Data/fiit/data/cll-para-sharing/raw_data/nli/nlpitu-xnli-42d713f6d3d5/data/snli_1.0_test_%s.txt' % lng
target_dir = '/media/piko/Data/fiit/data/cll-para-sharing/processed_data/nli'
output = ''
with codecs.open(nli_file, encoding='utf-8') as f:
    txts = []
    for line in f:
        o = line.split('\t')
        tag, premise, hypothesis = o[0], o[5], o[6] # 1,2 pre ine jazyky ako en
        premise = nltk.word_tokenize(premise)
        hypothesis = nltk.word_tokenize(hypothesis)
        hypothesis[0] = "%s\t%s" % (hypothesis[0], tag)
        lines = premise + hypothesis
        txt = '\n'.join(lines)
        txts.append(txt)

with codecs.open("%s/%s/dev" % (target_dir, lng), 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(txts))