import os
import glob
import codecs
import pickle
import numpy as np
from utils import general_utils as utils
from dataset import Dataset


class Cache:

    def __init__(self, config):
        self.config = config
        self.task_langs = []    # tuples (task, lang)
        self.lang_dicts = {}
        self.task_dicts = {}
        self.datasets = []
        self.embeddings = None

    def create(self, tasks=None, languages=None):
        self.delete()
        langs = set()
        if not tasks:
            tasks = utils.dirs(self.config.data_path)
        for task in tasks:
            folder_languages = utils.dirs(self.config.data_path+task+"/")
            if not languages:
                languages = folder_languages
            else:
                languages = [lang for lang in languages if lang in folder_languages]
            for lang in languages:
                langs.add(lang)
                self.task_langs.append((task, lang))

        emb_dicts = {}
        if self.config.word_emb_type == 'static':
            for lang in langs:
                emb_dicts[lang] = {'<unk>': np.zeros(self.config.word_emb_size)}
                emb_file = os.path.join(self.config.emb_path, lang)
                with codecs.open(emb_file) as f:
                    f.readline() # first init line in emb files
                    for line in f:
                        word, values = line.split(' ', 1)
                        values = np.array([float(val) for val in values.split(' ')])
                        emb_dicts[lang][word] = values
                        # TODO: check proper emb size

        for task, language in self.task_langs:
            self.lang_dicts.setdefault(language, {'<unk>'})
            self.task_dicts.setdefault(task, set())

            for sentence in self.load_files(task, language):
                for word in sentence[0]:
                    if word in emb_dicts[language]:
                        self.lang_dicts[language].add(word)
                for tag in sentence[1]:
                    self.task_dicts[task].add(tag)

        # Change to bidirectional hash (id_to_token, token_to_id)

        if 'ner' in tasks:
            self.task_dicts['ner'] = self.set_to_bidir_dict(self.task_dicts['ner'])

        if 'pos' in tasks:
            self.task_dicts['pos'] = self.set_to_bidir_dict(self.task_dicts['pos'])

        counter = 0
        for key in self.lang_dicts.keys():
            addition = len(self.lang_dicts[key])
            self.lang_dicts[key] = self.set_to_bidir_dict(self.lang_dicts[key], counter)
            counter += addition

        for task, lang in self.task_langs:
            for role in ['train', 'test', 'dev']:
                sentences = self.load_files(task, lang, role)
                for i in xrange(len(sentences)):
                    sentence_words, labels = sentences[i]
                    sentence_ids = []
                    for token in sentence_words:
                        if token in emb_dicts[lang]:
                            sentence_ids.append(self.lang_dicts[lang][1][token])
                        else:
                            sentence_ids.append(self.lang_dicts[lang][1]['<unk>'])
                    labels = [self.task_dicts[task][1][token] for token in labels]
                    sentences[i] = (np.array(sentence), np.array(labels))
                self.datasets.append(Dataset(lang, task, role, sentences))

        self.embeddings = np.zeros((counter, self.config.word_emb_size))
        for lang in langs:
            for id in self.lang_dicts[lang][0]:
                word = self.lang_dicts[lang][0][id]
                self.embeddings[id] = emb_dicts[lang][word]

        self.save()

    def set_to_bidir_dict(self, set, starting_id=0):
        set = list(set)
        id_to_token = {}
        token_to_id = {}
        for i in xrange(len(set)):
            i_ = i + starting_id
            id_to_token[i_] = set[i]
            token_to_id[set[i]] = i_
        return id_to_token, token_to_id

    def load_files(self, task, language, set_names=None):
        if set_names is None: set_names = ["train", "dev", "test"]
        if not isinstance(set_names, list): set_names = [set_names]
        sentences = []
        for set_name in set_names:
            with codecs.open(os.path.join(self.config.data_path, task, language, set_name), encoding='utf-8') as f:
                buffer = ([], [])
                for line in f:
                    if line.strip():
                        word, tag = line.split('\t')
                        buffer[0].append(word.strip())
                        buffer[1].append(tag.strip())
                    else:
                        sentences.append(buffer)
                        buffer = ([], [])
        return sentences

    def delete(self):
        files = glob.glob('%s*' % self.config.cache_path)
        for f in files:
            os.remove(f)

    def save(self):
        return pickle.dumps(self)


# TODO: pickling (kedy load, kedy create?)
#       opisat strukturu suboru, aby som ju mal poruke