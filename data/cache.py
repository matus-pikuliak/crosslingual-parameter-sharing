import os
import glob
import codecs
import pickle
from utils import general_utils as utils
from dataset import Dataset


class Cache:

    def __init__(self, config):
        self.config = config
        self.task_langs = []    # tuples (task, lang)
        self.lang_dicts = {}
        self.task_dicts = {}
        self.datasets = []

    def create(self, tasks=None, languages=None):
        self.delete()
        if not tasks:
            tasks = utils.dirs(self.config.data_path)
        for task in tasks:
            folder_languages = utils.dirs(self.config.data_path+task+"/")
            if not languages:
                languages = folder_languages
            else:
                languages = [lang for lang in languages if lang in folder_languages]
            for lang in languages:
                self.task_langs.append((task, lang))

        for task, language in self.task_langs:
            self.lang_dicts.setdefault(language, set())
            self.task_dicts.setdefault(task, set())

            for sentence in self.load_files(task, language):
                for word in sentence[0]:
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
                    sentence, labels = sentences[i]
                    sentence = [self.lang_dicts[lang][1][token] for token in sentence]
                    labels = [self.task_dicts[task][1][token] for token in labels]
                    # TODO np array? radsej ho urobit raz ako zakazdym pri davkovani
                    sentences[i] = (sentence, labels)
                self.datasets.append(Dataset(lang, task, role, sentences))

        #vyries embeddingy

        pck = self.save()
        return pck

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

    # def load(self, name):
    #     podla mena nacitaj pickle objekt

    # def decode_sentence(word_ids, label, lang, task)

    # def fetch_dataset(lang, task, role)
    # najdi v dostupnych ten pravy
    # zacachuj to do nejakeho pekneho hashu, nech to netreba zakazdym prehladavat vsetko

# TODO: dataset v np.array
#       embeddingy (zatial staticke)
#       pickling (kedy load, kedy create?)
#       opisat strukturu suboru, aby som ju mal poruke

# Cache =
# Dicts
# Embeddings - ordered set of embeddings copying ids of words, unique set for each language
# Data - each dataset transformed into easy to use format with ids instead of words