import os
import glob
from utils import general_utils as utils

class Cache:

    def __init__(self, config):
        self.config = config
        self.task_langs = []  # tuples (task, lang)

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

        for task, lang in self.task_langs:
            self.add_to_dictionary(task, lang)

        # combine dictionaries

        for task, lang in self.task_langs:
            self.process_data(task, lang)

        #vyries embeddingy

        self.save()


    def process_dictionary(self, task, language):
        zavolaj relevantny task procesor
        v task procesore sa postupne ako sa prechadzaju data do relevantneho objektu dictionary pridavaju nove zaznamy

    def process_data(self, task, language):
        return 0


    def delete(self):
        files = glob.glob('%s*' % self.config.cache_path)
        for f in files:
            os.remove(f)

    def save(self):
        zober data, dictionaries, embeddingy a nejako to uloz

# Cache =
# Dictionary - words + ids for every language
# Tag dictionary
# Embeddings - ordered set of embeddings copying ids of words, unique set for each language
# dict+emb mozno globalne pre vsetky jazyky
# Data - each dataset transformed into easy to use format with ids instead of words