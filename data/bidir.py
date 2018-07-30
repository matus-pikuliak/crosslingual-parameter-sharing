class Bidir():

    def __init__(self, st):
        self.token_to_id = dict()
        self.id_to_token = dict()

        for i, el in enumerate(st):
            self.token_to_id[el] = i
            self.id_to_token[i] = el

    def __len__(self):
        return len(self.token_to_id)