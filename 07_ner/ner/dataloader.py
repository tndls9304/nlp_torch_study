import torch
import numpy as np

from torchtext.legacy import data
from torchtext.legacy.datasets import TranslationDataset


class NERDataset:
    def __init__(self, config, w2v_stoi, w2v_vectors, device):
        self.config = config
        self.w2v_stoi = w2v_stoi
        self.w2v_vectors = w2v_vectors
        print(self.w2v_vectors.shape)
        self.device = device
        self.SRC = data.Field(tokenize=lambda x: x.split(),
                              unk_token='<unk>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True,
                              include_lengths=True)
        self.TRG = data.Field(tokenize=lambda x: x.split(),
                              unk_token='<unk>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True,
                              )
        self.train_data = TranslationDataset(path='dataset/klue-ner-v1_train_cleaned_tokenized',
                                             exts=('.src', '.trg'), fields=(self.SRC, self.TRG))
        self.test_data = TranslationDataset(path='dataset/klue-ner-v1_dev_cleaned_tokenized',
                                            exts=('.src', '.trg'), fields=(self.SRC, self.TRG))

        self.build_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator = data.BucketIterator(self.train_data, batch_size=self.config['batch_size'], device=device,
                                                  sort_key=lambda x: len(x.src), sort_within_batch=True)
        self.test_iterator = data.BucketIterator(self.test_data, batch_size=self.config['batch_size'], device=device,
                                                 sort_key=lambda x: len(x.src), sort_within_batch=True)

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        # self.SRC.vocab.set_vectors(self.w2v_stoi, torch.FloatTensor(self.w2v_vectors), 200)
        self.TRG.build_vocab(self.train_data, min_freq=1)
        print(f"Unique tokens in source vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target vocabulary: {len(self.TRG.vocab)}")
        print(self.TRG.vocab.stoi)
