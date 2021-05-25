import torchtext
if torchtext.__version__ == '0.6.0':
    from torchtext.legacy import data
else:
    from torchtext import data


class CLSDataset:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        # corpus_separator(filepath)
        self.title = data.Field(tokenize=lambda x: x.split(' '),
                                lower=True,
                                batch_first=True,
                                include_lengths=True)
        self.label = data.Field(lower=True,
                                batch_first=True)
        fields = [('label', self.label), ('title', self.title)]
        self.train_data, self.valid_data, self.test_data = data.TabularDataset.splits(path=self.config['cls_dir_path'],
                                                                                      train='train_tokenized.ynat',
                                                                                      validation='val_tokenized.ynat',
                                                                                      test='test_tokenized.ynat',
                                                                                      format='tsv',
                                                                                      fields=fields)

        self.build_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of valid data : {}'.format(len(self.valid_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data), sort=True, sort_within_batch=True,
            batch_size=self.config['cls_batch_size'], device=self.device, sort_key=lambda x: len(x.title))

    def build_vocab(self):
        self.title.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.label.build_vocab(self.train_data, min_freq=1)
        print(f"Unique tokens in title vocabulary: {len(self.title.vocab)}")
        print(f"Unique tokens in label vocabulary: {len(self.label.vocab)}")
