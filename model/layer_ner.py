import numpy as np

import constants
import utils.general_utils as utils
from model.layer_sqt import SQTLayer


class NERLayer(SQTLayer):

    def metrics_accumulator(self):
        o_tag = self.model.dl.task_vocabs['ner'].t2id[constants.NER_O_TAG]
        b_tags = [key for key in self.model.dl.task_vocabs['ner'].t2id.keys() if key.startswith('B-')]
        i_tags = [f'I-{b_tag[2:]}' for b_tag in b_tags]
        b_tags = [self.model.dl.task_vocabs['ner'].t2id[tag] for tag in b_tags]
        i_tags = [self.model.dl.task_vocabs['ner'].t2id[tag] for tag in i_tags]

        # for counting any tags
        correct = 0
        total = 1

        # for counting ner tags
        correct_ner = 0
        desired_ner = 1
        predicted_ner = 1

        # for counting ner chunks
        correct_chunks = 0
        desired_chunk_count = 1
        predicted_chunk_count = 1

        def output():
            out = {
                'acc': 100*correct/total,
                'tag_precision': correct_ner / predicted_ner,
                'tag_recall': correct_ner / desired_ner,
                'chunk_precision': correct_chunks / predicted_chunk_count,
                'chunk_recall': correct_chunks / desired_chunk_count
            }
            out['tag_f1'] = utils.f1(out['tag_precision'], out['tag_recall'])
            out['chunk_f1'] = utils.f1(out['chunk_precision'], out['chunk_recall'])
            return out

        while True:
            predicted, desired = (yield output())
            assert (len(predicted) == len(desired))

            # any tags
            total += len(predicted)
            correct += np.sum(predicted == desired)

            # ner tags
            correct_ner += np.sum(
                             np.logical_and(
                               (predicted == desired),
                               (predicted != o_tag)))
            desired_ner += np.sum(desired != o_tag)
            predicted_ner += np.sum(predicted != o_tag)

            # ner chunks
            predicted_chunks = self.chunks(predicted, b_tags, i_tags)
            desired_chunks = self.chunks(desired, b_tags, i_tags)
            desired_chunk_count += len(desired_chunks)
            predicted_chunk_count += len(predicted_chunks)
            correct_chunks += len(desired_chunks.intersection(predicted_chunks))

    @staticmethod
    def chunks(sequence, b_tags, i_tags):
        chunks = set()
        for i, id in enumerate(sequence):
            if id in b_tags:
                i_tag = i_tags[b_tags.index(id)]
                pointer = i + 1
                try:
                    while sequence[pointer] == i_tag:
                        pointer += 1
                except IndexError:
                    pass
                chunks.add((i, pointer - 1, id))
        return chunks


