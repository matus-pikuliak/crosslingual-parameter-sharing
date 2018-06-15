

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
                        # word_lengths.append(len(token))
                        # char_id_lists.append(self.pad_word(token, self.character_dict))
                    label_ids = [self.token_to_id[task][token] for token in labels]
                    samples[i] = (np.array(word_ids, dtype=np.int32), np.array(label_ids, dtype=np.int32), len(words), np.array(char_id_lists, dtype=np.int32), np.array(word_lengths, dtype=np.int32))
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
        token = list(token)
        token = token[:max_length]
        for i, character in enumerate(token):
            if character not in character_dict:
                token[i] = self.__class__.unk_char_token
        return np.array([character_dict[character] for character in token])