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
        valid_languages = set()
        if not tasks:
            tasks = utils.dirs(self.config.data_path)
        for task in tasks:
            folder_languages = utils.dirs(self.config.data_path+task+"/")
            if not languages:
                languages = folder_languages
            else:
                languages = [lang for lang in languages if lang in folder_languages]
            for lang in languages:
                valid_languages.add(lang)
                self.task_langs.append((task, lang))

        emb_dicts = {}
        if self.config.word_emb_type == 'static':
            for lang in valid_languages:
                emb_dicts[lang] = {'<unk>': np.zeros(self.config.word_emb_size)}
                emb_file = os.path.join(self.config.emb_path, lang)
                with codecs.open(emb_file) as f:
                    f.readline()  # first init line in emb files
                    for line in f:
                        word, values = line.split(' ', 1)
                        values = np.array([float(val) for val in values.split(' ')])
                        emb_dicts[lang][word] = values
                        # TODO: check proper emb size

        for task, language in self.task_langs:
            self.lang_dicts.setdefault(language, {'<unk>'})
            self.task_dicts.setdefault(task, set())

            for (words, labels, _) in self.load_files(task, language):
                for word in words:
                    if word in emb_dicts[language]:
                        self.lang_dicts[language].add(word)
                for tag in labels:
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
                samples = self.load_files(task, lang, role)
                for i, (sen_words, sen_labels, length) in enumerate(samples):
                    sen_ids = []
                    for token in sen_words:
                        if token in emb_dicts[lang]:
                            sen_ids.append(self.lang_dicts[lang][1][token])
                        else:
                            sen_ids.append(self.lang_dicts[lang][1]['<unk>'])
                    label_ids = [self.task_dicts[task][1][token] for token in sen_labels]
                    samples[i] = (np.array(sen_ids), np.array(label_ids), length)
                self.datasets.append(Dataset(lang, task, role, np.array(samples)))

        self.embeddings = np.zeros((counter, self.config.word_emb_size))
        for lang in valid_languages:
            for id in self.lang_dicts[lang][0]:
                word = self.lang_dicts[lang][0][id]
                self.embeddings[id] = emb_dicts[lang][word]

        #self.save()

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
        samples = []
        for set_name in set_names:
            with codecs.open(os.path.join(self.config.data_path, task, language, set_name), encoding='utf-8') as f:
                word_buffer = []
                label_buffer = []
                for line in f:
                    if line.strip():
                        word, tag = line.split('\t')
                        word_buffer.append(word.strip())
                        label_buffer.append(tag.strip())
                    else:
                        samples.append((word_buffer, label_buffer, len(word_buffer)))
                        word_buffer = []
                        label_buffer = []
        return samples

    def delete(self):
        files = glob.glob('%s*' % self.config.cache_path)
        for f in files:
            os.remove(f)

    def save(self):
        return pickle.dumps(self)

    def fetch_dataset(self, language, task, role):
        for dataset in self.datasets:
            if language == dataset.language and task == dataset.task and role == dataset.role: return dataset
        raise 'No dataset with required parameters'

# TODO: pickling (kedy load, kedy create?)
#       opisat strukturu suboru, aby som ju mal poruke