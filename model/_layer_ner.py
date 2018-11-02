import numpy as np

import constants
from model._layer_sqt import SQTLayer


class NERLayer(SQTLayer):

    def metrics_accumulator(self):
        o_tag = self.model.dl.task_vocabs['ner'].t2id[constants.NER_O_TAG]
        total = 1
        correct = 0

        def output():
            return {'acc': 100*correct/total}

        while True:
            predicted, desired = (yield output())
            assert (len(predicted) == len(desired))
            total += len(predicted)
            correct += np.sum(predicted == desired)