import random
import torch

from torchtext import data
from torchtext.datasets import Multi30k


class Seq2SeqDataset:
    def __init__(self, config, tokenize_src, tokenize_trg, device):
        self.config = config
        self.device = device
        self.SRC = data.Field(tokenize=tokenize_src,
                              init_token=None,
                              eos_token='<eos>',
                              pad_token='<pad>',
                              unk_token='<unk>',
                              lower=True,
                              include_lengths=True,
                              batch_first=True,
                              preprocessing=lambda x: x[::-1])  # reversing source sentence
        self.TRG = data.Field(tokenize=tokenize_trg,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              unk_token='<unk>',
                              lower=True,
                              include_lengths=True,
                              batch_first=True)
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=(config['src_ext'], config['trg_ext']),
                                                                           fields=(self.SRC, self.TRG))
        self.build_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of valid data : {}'.format(len(self.valid_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.config['batch_size'], device=self.device, sort_within_batch=True, sort_key=lambda x: len(x.src))

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.TRG.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")
