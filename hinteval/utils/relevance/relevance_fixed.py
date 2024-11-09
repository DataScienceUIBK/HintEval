import torch
import os
import numpy as np
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from hinteval.utils.functions.download_manager import EmbeddingsDownloader


class UnknownWordVecCache(object):
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


class Batch:
    def __init__(self, stop_words, device, glove_version, force_download, unk_init):
        self.TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)
        self.EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True,
                                     tokenize=lambda x: x)
        self.fields = [('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD),
                       ('ext_feats', self.EXT_FEATS_FIELD)]
        self.stop_words = stop_words
        vectors_dir = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'embeddings')
        if glove_version == 'glove.6B':
            vectors_name = 'glove.6B.300d.txt'
        else:
            vectors_name = 'glove.42B.300d.txt'
        EmbeddingsDownloader.download(glove_version, force_download)
        self.vectors = Vectors(name=vectors_name, cache=vectors_dir, unk_init=unk_init)
        self.TEXT_FIELD.build_vocab(vectors=self.vectors)
        self.device = device

    def _compute_overlap_features(self, questions, answers, with_stoplist):
        stoplist = self.stop_words if with_stoplist else []
        feats_overlap = []
        for question, answer in zip(questions, answers):
            q_set = set([q for q in question if q not in stoplist])
            a_set = set([a for a in answer if a not in stoplist])
            word_overlap = q_set.intersection(a_set)
            if len(q_set) == 0 and len(a_set) == 0:
                overlap = 0
            else:
                overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

            feats_overlap.append(np.array([overlap]))
        return np.array(feats_overlap)

    def get_iterator(self, sentences_1, sentences_2, batch_size):

        examples = []
        sent_list_1 = [l.strip().split(' ') for l in sentences_1]
        sent_list_2 = [l.strip().split(' ') for l in sentences_2]

        overlap_feats = self._compute_overlap_features(sentences_1, sentences_2, with_stoplist=False)
        overlap_feats_stoplist = self._compute_overlap_features(sentences_1, sentences_2, with_stoplist=True)
        overlap_feats = np.hstack(
            [overlap_feats, overlap_feats_stoplist, overlap_feats, overlap_feats_stoplist]).tolist()

        for i, (l1, l2, ext_feats) in enumerate(zip(sent_list_1, sent_list_2, overlap_feats)):
            example_list = [l1, l2, ext_feats]
            example = Example.fromlist(example_list, self.fields)
            examples.append(example)

        batch = Dataset(examples, self.fields)

        iterator = BucketIterator(batch, batch_size=batch_size, repeat=False, device=self.device, shuffle=False)
        return iterator
