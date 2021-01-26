import random
import torch

from torchtext import data
from gensim.models import KeyedVectors

from cnn.utils import simple_reader, simple_writer, clean_list


class CNNDataset:
    def __init__(self, conf_dict, device):
        self.positive_path = conf_dict['POS_PATH']
        self.negative_path = conf_dict['NEG_PATH']
        self.cleaned_path = conf_dict['CL_PATH']
        self.max_length = conf_dict['MAX_LEN']
        self.w2v_path = conf_dict['W2V_PATH']
        self.vocab_size = conf_dict['VOCAB_SIZE']
        self.batch_size = conf_dict['BATCH_SIZE']
        self.seed = conf_dict['SEED']
        self.test_ratio = conf_dict['TEST_RATIO']
        self.valid_ratio = conf_dict['VALID_RATIO']
        self.device = device

        # define fields
        self.TEXT = data.Field(sequential=True, use_vocab=True, tokenize=lambda x: x.split(' '), lower=False,
                               batch_first=True, fix_length=self.max_length)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # preprocessing corpus
        self.clean_data()

        # load dataset with torchtext TabularDataset
        self.dataset = data.TabularDataset(path=self.cleaned_path, fields=[('label', self.LABEL), ('text', self.TEXT)],
                                           format='csv', csv_reader_params={'delimiter': '\t'}, skip_header=True)
        # load w2v
        self.word2vec = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        self.word2index = {token: token_index for token_index, token in enumerate(self.word2vec.index2word)}

        # build_vocab
        self.build_vocab()

    def clean_data(self):
        data_list = simple_reader(self.positive_path) + simple_reader(self.negative_path)
        data_list = clean_list(data_list)
        simple_writer(self.cleaned_path, data_list)

    def build_vocab(self):
        train_data, test_data = self.dataset.split(split_ratio=1-self.test_ratio, random_state=random.seed(self.seed))
        train_data, valid_data = train_data.split(split_ratio=1-(self.valid_ratio*10/((1-self.test_ratio)*10)), random_state=random.seed(self.seed))

        self.TEXT.build_vocab(train_data, max_size=self.vocab_size)
        self.LABEL.build_vocab(train_data)
        self.TEXT.vocab.set_vectors(self.word2index, torch.from_numpy(self.word2vec.vectors).float(), self.word2vec.vector_size)

        self.train_iterator = data.Iterator(train_data, batch_size=self.batch_size, train=True, device=self.device, sort_key=lambda x: len(x.text), sort_within_batch=False)
        self.valid_iterator = data.Iterator(valid_data, batch_size=self.batch_size, train=False, device=self.device, sort_key=lambda x: len(x.text), sort_within_batch=False)
        self.test_iterator = data.Iterator(test_data, batch_size=self.batch_size, train=False, device=self.device, sort_key=lambda x: len(x.text), sort_within_batch=False)

        print('number of training data : {}'.format(len(train_data)))
        print('number of valid data : {}'.format(len(valid_data)))
        print('number of test data : {}'.format(len(test_data)))


