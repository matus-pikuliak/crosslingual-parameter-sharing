from paths import paths


class Config:

    def __init__(self):
        for k in paths:
            setattr(self, k, paths[k])
