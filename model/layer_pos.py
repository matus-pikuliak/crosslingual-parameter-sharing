import numpy as np

from model.layer_sqt import SQTLayer


class POSLayer(SQTLayer):

    def metrics_accumulator(self):
        total = 1
        correct = 0

        def output():
            return {'acc': 100*correct/total}

        while True:
            predicted, desired = (yield output())
            assert (len(predicted) == len(desired))
            total += len(predicted)
            correct += np.sum(predicted == desired)
