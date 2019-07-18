import numpy as np

from model.model_sqt import ModelSQT


class ModelPOS(ModelSQT):

    def metric_names(self):
        return ['correct_tags']

    def metrics_from_batch(self, logits, desired, lengths, transition_params):
        return sum(
            np.sum(predicted == desired)
            for predicted, desired
            in self.crf_predict(logits, desired, lengths, transition_params)
        )

    def evaluate_task(self, results):
        return {
            'acc': 100 * sum(results['correct_tags']) / sum(results['length'])
        }
