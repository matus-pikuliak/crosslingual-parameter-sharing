import numpy as np

import constants
import utils.general_utils as utils
from model._layer_sqt import SQTLayer


class NERLayer(SQTLayer):

    def metrics_accumulator(self):
        o_tag = self.model.dl.task_vocabs['ner'].t2id[constants.NER_O_TAG]
        total = 1
        correct = 0
        correct_ner = 0
        desired_ner = 1
        predicted_ner = 1

        def output():
            out = {
                'acc': 100*correct/total,
                'tag_precision': correct_ner / predicted_ner,
                'tag_recall': correct_ner / desired_ner
            }
            out['tag_f1'] = utils.f1(out['tag_precision'], out['tag_recall'])
            return out

        while True:
            predicted, desired = (yield output())
            assert (len(predicted) == len(desired))

            total += len(predicted)
            correct += np.sum(predicted == desired)
            correct_ner += np.sum(
                             np.logical_and(
                               (predicted == desired),
                               (predicted != o_tag)))
            desired_ner += np.sum(desired != o_tag)
            predicted_ner += np.sum(predicted != o_tag)