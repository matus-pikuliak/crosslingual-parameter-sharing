class Bidir():

    def __init__(self, st, start=0):
        self.token_to_id = dict()
        self.id_to_token = dict()

        for i, el in enumerate(st):
            self.token_to_id[el] = i+start
            self.id_to_token[i+start] = el
