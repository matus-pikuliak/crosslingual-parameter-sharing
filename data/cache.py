import os
import glob
import codecs
import pickle
import numpy as np
from utils import general_utils as utils
from dataset import Dataset


class Cache:
    '''Cache object contains following parts:
    config - Config file with the information about various configuration settings that were used to create this cache file.
    task_langs - list of (task, language) tuples of present datasets
    lang_dicts - dictionaries of words for individual languages
    task_dicts - dictionaries of tags for individual tasks
    datasets - Dataset objects for individual datasets
    embeddings - cached embeddings for all the words
    '''
    unk_word_token = '<unk>'
    unk_char_token = u'\u2716'
    empty_char_token = u'\u2717'

    def __init__(self, config):
        self.config = config
        self.task_langs = []    # tuples (task, lang)
        self.datasets = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.lang_dicts = {}
        self.task_dicts = {}
        self.character_dict = {}
        self.embeddings = None

    def get_task_langs(self, tasks, languages):
        task_langs = []
        for task in tasks:
            folder_languages = list(utils.dirs(self.config.data_path+task+"/"))
            if not languages:
                languages = folder_languages
            else:
                languages = [lang for lang in languages if lang in folder_languages]
            for lang in languages:
                task_langs.append((task, lang))
        # TODO: Vyhod chybu ak sa meno jazyka a ulohy rovna.
        return task_langs

    def valid_languages(self, task_langs):
        return set([lang for _, lang in task_langs])

    def load_embeddings(self, task_langs):
        '''
        Return a dict:
        {
            'lang1': {'word1': np.array([d1, d2, d3]), 'word2' ... },
            'lang2': ...
        }
        '''
        embeddings = {}
        if self.config.word_emb_type == 'static':
            for lang in self.valid_languages(task_langs):
                # zero vector for unknown words
                embeddings[lang] = {'<unk>': np.zeros(self.config.word_emb_size)}
                emb_file = os.path.join(self.config.emb_path, lang)
                with codecs.open(emb_file, encoding='utf-8') as f:
                    f.readline()  # first init line in emb files
                    for line in f:
                        try:
                            word, values = line.split(' ', 1)
                            values = np.array([float(val) for val in values.split(' ')])
                            embeddings[lang][word] = values
                        except:
                            print("Warning: there is a ill formatted line for language %s: '%s'" % (lang, line))
        return embeddings

    def generate_dicts(self, task_langs, embeddings):
        lang_dicts = {}
        task_dicts = {}
        character_dict = {}

        for task, language in task_langs:
            lang_dicts.setdefault(language, {'<unk>'}) # Unkown word token for each language
            task_dicts.setdefault(task, set())

            for (words, labels) in self.load_files(task, language):
                for word in words:
                    if word.lower() in embeddings[language]:
                        lang_dicts[language].add(word)
                    for character in word:
                        if character not in character_dict:
                            character_dict[character] = 0
                        character_dict[character] += 1
                for tag in labels:
                    task_dicts[task].add(tag)

        character_dict = set([character for character in character_dict if character_dict[character] > 30])
        character_dict.add(self.__class__.unk_char_token)
        character_dict.add(self.__class__.empty_char_token)
        character_dict = self.set_to_bidir_dict(character_dict)[1]

        return lang_dicts, task_dicts, character_dict

    def create(self, tasks=None, languages=None):
        if not tasks:
            tasks = utils.dirs(self.config.data_path)
        self.task_langs = self.get_task_langs(tasks, languages)

        _embeddings = self.load_embeddings(self.task_langs)

        self.lang_dicts, self.task_dicts, self.character_dict = self.generate_dicts(self.task_langs, _embeddings)

        # Change to bidirectional hash (id_to_token, token_to_id)
        if 'ner' in tasks:
            self.id_to_token['ner'], self.token_to_id['ner'] = self.set_to_bidir_dict(self.task_dicts['ner'])

        if 'pos' in tasks:
            self.id_to_token['pos'], self.token_to_id['pos'] = self.set_to_bidir_dict(self.task_dicts['pos'])

        counter = 0
        for lang in self.lang_dicts.keys():
            addition = len(self.lang_dicts[lang])
            self.id_to_token[lang], self.token_to_id[lang] = self.set_to_bidir_dict(self.lang_dicts[lang], counter)
            counter += addition

        for task, lang in self.task_langs:
            for role in ['train', 'test', 'dev']:
                samples = self.load_files(task, lang, role)
                for i, (words, labels) in enumerate(samples):
                    word_ids = []
                    char_id_lists = []
                    word_lengths = []
                    for token in words:
                        if token.lower() in _embeddings[lang]: # TODO: Tento lower je zradny
                            word_ids.append(self.token_to_id[lang][token])
                        else:
                            word_ids.append(self.token_to_id[lang]['<unk>'])
                        word_lengths.append(len(token))
                        char_id_lists.append(self.pad_word(token, self.character_dict))
                    label_ids = [self.token_to_id[task][token] for token in labels]
                    samples[i] = (np.array(word_ids), np.array(label_ids), len(words), np.array(char_id_lists), np.array(word_lengths))
                self.datasets.append(Dataset(lang, task, role, np.array(samples)))

        self.embeddings = np.zeros((counter, self.config.word_emb_size))
        for lang in self.valid_languages(self.task_langs):
            for id in self.id_to_token[lang]:
                word = self.id_to_token[lang][id]
                self.embeddings[id] = _embeddings[lang][word.lower()]

    def set_to_bidir_dict(self, set, starting_id=0):
        set = list(set)
        id_to_token = {}
        token_to_id = {}
        for i in xrange(len(set)):
            i_ = i + starting_id
            id_to_token[i_] = set[i]
            token_to_id[set[i]] = i_
        return id_to_token, token_to_id

    def pad_word(self, token, character_dict):
        max_length = 30
        token = token.ljust(max_length, self.__class__.empty_char_token)
        token = list(token)
        token = token[:max_length]
        for i, character in enumerate(token):
            if character not in character_dict:
                token[i] = self.__class__.unk_char_token
        return np.array([character_dict[character] for character in token])


    def load_files(self, task, language, set_names=None):

        if set_names is None:
            set_names = ["train", "dev", "test"]

        if not isinstance(set_names, list):
            set_names = [set_names]

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
                        samples.append((word_buffer, label_buffer))
                        word_buffer = []
                        label_buffer = []
        return samples

    def delete(self):
        files = glob.glob('%s*' % self.config.cache_path)
        for f in files:
            os.remove(f)

    def save(self):
        self.delete()
        f = codecs.open('%scache.pckl' % self.config.cache_path, 'wb')
        return pickle.dump(self, f)

    def load(self):
        f = codecs.open('%scache.pckl' % self.config.cache_path, 'rb')
        return pickle.load(f)

    def fetch_dataset(self, task, language, role):
        for dataset in self.datasets:
            if language == dataset.language and task == dataset.task and role == dataset.role: return dataset
        raise BaseException('No dataset with required parameters: %s %s %s' % (language, task, role))
