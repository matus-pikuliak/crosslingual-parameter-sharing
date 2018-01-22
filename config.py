from paths import paths

class Config:

    def __init__(self):
        self.attributes = {}
        for k in paths:
            self[k] = paths[k]

    def __setitem__(self, key, value):
        self.attributes[key] = value

    def __getitem__(self, key):
        return self.attributes[key]
