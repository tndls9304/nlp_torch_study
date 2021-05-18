import codecs

from torchtext.legacy import data, datasets
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from general_utils.utils import pickle_reader


def corpus_separator(output_name):
    tokenizer = Mecab()
    corpus = pickle_reader('dataset/petitions_2019-01.p')
    writer_src_train = codecs.open(output_name + 'train.src', 'w', encoding='utf-8')
    writer_src_valid = codecs.open(output_name + 'val.src', 'w', encoding='utf-8')
    writer_src_test = codecs.open(output_name + 'test.src', 'w', encoding='utf-8')
    writer_trg_train = codecs.open(output_name + 'train.trg', 'w', encoding='utf-8')
    writer_trg_valid = codecs.open(output_name + 'val.trg', 'w', encoding='utf-8')
    writer_trg_test = codecs.open(output_name + 'test.trg', 'w', encoding='utf-8')
    train_lines, test_lines = train_test_split(corpus, test_size=0.05, random_state=1234)
    train_lines, valid_lines = train_test_split(train_lines, test_size=1/19, random_state=1234)
    for line in train_lines:
        tokens = tokenizer.morphs(line.strip())
        writer_src_train.write(' '.join(tokens[:-1]) + '\n')
        writer_trg_train.write(' '.join(tokens[1::]) + '\n')
    for line in valid_lines:
        tokens = tokenizer.morphs(line.strip())
        writer_src_valid.write(' '.join(tokens[:-1]) + '\n')
        writer_trg_valid.write(' '.join(tokens[1::]) + '\n')
    for line in test_lines:
        tokens = tokenizer.morphs(line.strip())
        writer_src_test.write(' '.join(tokens[:-1]) + '\n')
        writer_trg_test.write(' '.join(tokens[1::]) + '\n')

    writer_src_train.close()
    writer_src_valid.close()
    writer_src_test.close()
    writer_trg_train.close()
    writer_trg_valid.close()
    writer_trg_test.close()


class ELMODataset:
    def __init__(self, config, filepath, device):
        self.config = config
        self.device = device
        # corpus_separator(filepath)
        self.SRC = data.Field(tokenize=lambda x: x.split(' '),
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True,
                              include_lengths=True)
        self.TRG = data.Field(tokenize=lambda x: x.split(' '),
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.train_data, self.valid_data, self.test_data = \
            datasets.TranslationDataset.splits(path=filepath, exts=(self.config['src_ext'], self.config['trg_ext']),
                                               fields=(self.SRC, self.TRG))

        self.build_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of valid data : {}'.format(len(self.valid_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data), sort=True, sort_within_batch=True,
            batch_size=self.config['batch_size'], device=self.device)

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.TRG.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")
