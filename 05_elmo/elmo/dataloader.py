import codecs

from torchtext import data, datasets
from konlpy.tag import Mecab


def corpus_separator(input_file, output_name):
    tokenizer = Mecab()
    reader = codecs.open(input_file, 'r', encoding='utf-8')
    writer_src = codecs.open(output_name + '.src', 'w', encoding='utf-8')
    writer_trg = codecs.open(output_name + '.trg', 'w', encoding='utf-8')

    for line in reader:
        tokens = tokenizer.morphs(line.strip())
        writer_src.write(' '.join(tokens[:-1]) + '\n')
        writer_trg.write(' '.join(tokens[1::]) + '\n')

    reader.close()
    writer_src.close()
    writer_trg.close()


class ELMODataset:
    def __init__(self, config, filepath, tokenizer, device):
        self.config = config
        self.device = device
        self.SRC = data.Field(tokenize=tokenizer,
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.TRG = data.Field(tokenize=tokenizer,
                              init_token='<sos>',
                              eos_token='<eos>',
                              pad_token='<pad>',
                              lower=True,
                              batch_first=True)
        self.train_data, self.valid_data, self.test_data = \
            datasets.TranslationDataset.splits(path=filepath, exts=('.src', '.trg'),
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
