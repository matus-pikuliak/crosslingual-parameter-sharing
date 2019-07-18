import numpy as np

import constants
import utils.general_utils as utils
from model.model_sqt import ModelSQT


class ModelNER(ModelSQT):

    def __init__(self, *args):
        ModelSQT.__init__(self, *args)

        self.o_tag = self.orch.dl.task_vocabs['ner'].label_to_id(constants.NER_O_TAG)

        b_tags = [key for key in self.orch.dl.task_vocabs['ner'] if key.startswith('B-')]
        self.b_tags = [self.orch.dl.task_vocabs['ner'].label_to_id(tag) for tag in b_tags]

        i_tags = [f'I-{b_tag[2:]}' for b_tag in b_tags]
        self.i_tags = [self.orch.dl.task_vocabs['ner'].label_to_id(tag) for tag in i_tags]

    def metric_names(self):
        return [
            'correct_tags',
            'correct_ner', 'desired_ner', 'predicted_ner',
            'correct_chunk_count', 'desired_chunk_count', 'predicted_chunk_count'
        ]

    def metrics_from_batch(self, logits, desired, lengths, transition_params):

        # for counting any tags
        correct_tags = 0

        # for counting ner tags
        correct_ner = 0
        desired_ner = 0
        predicted_ner = 0

        # for counting ner chunks
        correct_chunks = 0
        desired_chunk_count = 0
        predicted_chunk_count = 0

        for predicted, desired in self.crf_predict(logits, desired, lengths, transition_params):

            # any tags
            correct_tags += np.sum(predicted == desired)

            # ner tags
            correct_ner += np.sum(
                             np.logical_and(
                               (predicted == desired),
                               (predicted != self.o_tag)))
            desired_ner += np.sum(desired != self.o_tag)
            predicted_ner += np.sum(predicted != self.o_tag)

            # ner chunks
            predicted_chunks = self.chunks(predicted)
            desired_chunks = self.chunks(desired)
            desired_chunk_count += len(desired_chunks)
            predicted_chunk_count += len(predicted_chunks)
            correct_chunks += len(desired_chunks.intersection(predicted_chunks))

        return [
            correct_tags,
            correct_ner, desired_ner, predicted_ner,
            correct_chunks, desired_chunk_count, predicted_chunk_count
        ]

    def evaluate_task(self, results):

        for metric in self.metric_names():
            results[metric] = sum(results[metric])

        output = {
            'acc': 100 * results['correct_tags'] / sum(results['length']),
            'tag_precision': results['correct_ner'] / results['predicted_ner'],
            'tag_recall': results['correct_ner'] / results['desired_ner'],
            'chunk_precision': results['correct_chunk_count'] / results['predicted_chunk_count'],
            'chunk_recall': results['correct_chunk_count'] / results['desired_chunk_count']
        }

        output.update({
            'tag_f1': utils.f1(output['tag_precision'], output['tag_recall']),
            'chunk_f1': utils.f1(output['chunk_precision'], output['chunk_recall'])
        })

        return output

    def chunks(self, sequence):
        chunks = set()
        for i, id in enumerate(sequence):
            if id in self.b_tags:
                i_tag = self.i_tags[self.b_tags.index(id)]
                pointer = i + 1
                try:
                    while sequence[pointer] == i_tag:
                        pointer += 1
                except IndexError:
                    pass
                chunks.add((i, pointer - 1, id))
        return chunks


