import codecs

import torchtext
if torchtext.__version__ == '0.6.0':
    from torchtext.legacy import data, datasets
else:
    from torchtext import data, datasets

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


class ELMOSample:
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.examples = list()
        self.iterator = None
        self.tokenizer = Mecab()
        self.SRC = data.Field(tokenize=lambda x: x.split(' '),
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True,
                              include_lengths=True)
        self.rSRC = data.Field(tokenize=lambda x: x.split(' '),
                               eos_token='<eos>',
                               pad_token='<pad>',
                               lower=True,
                               batch_first=True,
                               include_lengths=True,
                               preprocessing=lambda x: x[::-1])

        self.SRC.vocab = pickle_reader(self.config['src_field_path'])
        self.rSRC.vocab = self.SRC.vocab

    def sent2iterator(self, sentence):
        self.sentencelist2iterator([sentence])

    def sent2example(self, sentence):
        sentence_dict = {'src': sentence, 'rsrc': sentence}
        example = data.Example.fromdict(sentence_dict, fields={'src': ('src', self.SRC), 'rsrc': ('rsrc', self.rSRC)})
        return example

    def sentencelist2iterator(self, sentences):
        examples = list()
        for sentence in sentences:
            example = self.sent2example(sentence)
            examples.append(example)
        dataset = data.Dataset(examples, fields=[('src', self.SRC), ('rsrc', self.rSRC)])
        self.iterator = data.Iterator(dataset, batch_size=1, sort_key=lambda x: len(x.src), sort=True,
                                      sort_within_batch=True, device=self.device)


class ELMODataset:
    def __init__(self, config, build_vocab=True):
        self.config = config
        self.device = self.config['device']
        # corpus_separator(filepath)
        if config.get('max_length', 0) == 0:
            self.SRC = data.Field(tokenize=lambda x: x.split(' '),
                                  eos_token='<eos>',
                                  pad_token='<pad>',
                                  lower=True,
                                  batch_first=True,
                                  include_lengths=True),
            self.rSRC = data.Field(tokenize=lambda x: x.split(' '),
                                   eos_token='<eos>',
                                   pad_token='<pad>',
                                   lower=True,
                                   batch_first=True,
                                   include_lengths=True,
                                   preprocessing=lambda x: x[::-1])
            self.TRG = data.Field(tokenize=lambda x: x.split(' '),
                                  eos_token='<eos>',
                                  pad_token='<pad>',
                                  lower=True,
                                  batch_first=True,
                                  is_target=True)
            self.rTRG = data.Field(tokenize=lambda x: x.split(' '),
                                   eos_token='<eos>',
                                   pad_token='<pad>',
                                   lower=True,
                                   batch_first=True,
                                   is_target=True,
                                   preprocessing=lambda x: x[::-1])
        else:
            self.SRC = data.Field(tokenize=lambda x: x.split(' '),
                                  eos_token='<eos>',
                                  pad_token='<pad>',
                                  lower=True,
                                  batch_first=True,
                                  include_lengths=True,
                                  preprocessing=lambda x: x[:config['max_length']])
            self.rSRC = data.Field(tokenize=lambda x: x.split(' '),
                                   eos_token='<eos>',
                                   pad_token='<pad>',
                                   lower=True,
                                   batch_first=True,
                                   include_lengths=True,
                                   preprocessing=lambda x: x[::-1][:config['max_length']])
            self.TRG = data.Field(tokenize=lambda x: x.split(' '),
                                  eos_token='<eos>',
                                  pad_token='<pad>',
                                  lower=True,
                                  batch_first=True,
                                  is_target=True,
                                  preprocessing=lambda x: x[:config['max_length']+1])
            self.rTRG = data.Field(tokenize=lambda x: x.split(' '),
                                   eos_token='<eos>',
                                   pad_token='<pad>',
                                   lower=True,
                                   batch_first=True,
                                   is_target=True,
                                   preprocessing=lambda x: x[::-1][:config['max_length']+1])

        self.train_data, self.valid_data, self.test_data = \
            data.TabularDataset.splits(path='elmo/dataset/', train='train.data2', validation='val.data2', test='test.data2',
                                       fields=[('src', self.SRC), ('rsrc', self.rSRC), ('trg', self.TRG), ('rtrg', self.rTRG)], format='tsv')
        """
        self.train_data, self.valid_data, self.test_data = \
            datasets.TranslationDataset.splits(path=config['filepath'], exts=(self.config['src_ext'], self.config['trg_ext']),
                                               fields=(self.SRC, self.TRG))
        """
        if build_vocab:
            self.build_vocab()
        else:
            self.load_vocab()

        print('number of training data : {}'.format(len(self.train_data)))
        print('number of valid data : {}'.format(len(self.valid_data)))
        print('number of test data : {}'.format(len(self.test_data)))

        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data), sort=True, sort_within_batch=True,
            batch_size=self.config['batch_size'], device=self.device, sort_key=lambda x: len(x.src))

    def build_vocab(self):
        self.SRC.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.rSRC.vocab = self.SRC.vocab
        self.TRG.build_vocab(self.train_data, min_freq=self.config['min_freq'])
        self.rTRG.vocab = self.TRG.vocab
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")

    def load_vocab(self):
        self.SRC.vocab = pickle_reader(self.config['src_field_path'])
        self.TRG.vocab = pickle_reader(self.config['trg_field_path'])
        print(f"Unique tokens in source {self.config['src_ext'][1:]} vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target {self.config['trg_ext'][1:]} vocabulary: {len(self.TRG.vocab)}")
