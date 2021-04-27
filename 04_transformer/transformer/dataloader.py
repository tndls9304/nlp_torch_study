from torchtext.legacy import data, datasets
from torchtext.legacy.datasets import Multi30k


class TransformerDataset:
    def __init__(self, config, tokenize_src, tokenize_trg, device):
        self.config = config
        self.device = device
        self.SRC = data.Field(tokenize=tokenize_src,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.TRG = data.Field(tokenize=tokenize_trg,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True,
                              )
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=(config['src_ext'], config['trg_ext']),
                                                                           fields=(self.SRC, self.TRG))
        self.build_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of valid data : {}'.format(len(self.valid_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.config['batch_size'], device=self.device)

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.TRG.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")


class DialogDataset:
    def __init__(self, config, filepath, tokenize, device):
        self.config = config
        self.device = device
        self.SRC = data.Field(tokenize=tokenize,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.TRG = data.Field(tokenize=tokenize,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.train_data, self.valid_data, self.test_data = \
            datasets.TranslationDataset.splits(path=filepath, exts=('.src', '.trg'),
                                               fields=(self.SRC, self.TRG))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.config['batch_size'], device=self.device)

        self.build_vocab()

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.TRG.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")
