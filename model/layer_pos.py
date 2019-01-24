import numpy as np

from model.layer_sqt import SQTLayer


class POSLayer(SQTLayer):

    def metric_names(self):
        return ['correct_tags']

    def metrics_from_batch(self, logits, desired, lengths, transition_params):
        return sum(
            np.sum(predicted == desired)
            for predicted, desired
            in self.crf_predict(logits, desired, lengths, transition_params)
        )

    def evaluate(self, iterator, dataset):

        fetches = self.basic_fetches()
        fetches.update(self.metrics)
        results = self.evaluate_batches(iterator, dataset, fetches)

        output = {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
            'acc': 100 * sum(results['correct_tags']) / sum(results['length']),
            'oi': sum(results['unit_to_unit_influence'])
        } # TODO: proper metrics for each task (share most computation in layer.method)

        return output
