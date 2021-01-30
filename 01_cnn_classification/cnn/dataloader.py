import random
import torch

import numpy as np

from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors

from cnn.utils import simple_reader, simple_writer, clean_list


class CNNDataset(Dataset):
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
        self.embedding_dim = conf_dict['EMBEDDING_DIM']
        self.device = device

        self.corpus = list()
        self.labels = list()
        self.vectors = list()
        self.word2vec = None
        self.word2index = None
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.stoi = None
        self.itos = None

        self._load_dataset()
        self._load_word2vec()

    def _load_dataset(self):
        positive_corpus = simple_reader(self.positive_path)
        negative_corpus = simple_reader(self.negative_path)

        self.corpus = clean_list(positive_corpus) + clean_list(negative_corpus)
        self.labels = ([1] * len(positive_corpus)) + ([0] * len(negative_corpus))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]

    def _load_word2vec(self):
        # load w2v
        self.word2vec = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        self.word2index = {token: token_index for token_index, token in enumerate(self.word2vec.index2word)}

    def build_vocab(self, train_indices):
        train_corpus = [self.corpus[i] for i in train_indices]
        counter = self._generate_freq_vocab(train_corpus)
        counter = counter.copy()

        self.itos = [self.pad_token, self.unk_token]

        for tok in [self.pad_token, self.unk_token]:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if len(self.itos) == self.vocab_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self._build_embeddings()

    def _build_embeddings(self):
        value = random.randint(0, len(self.stoi))
        a = np.var(self.word2vec.wv.vectors)
        for word in self.stoi:
            if word in self.word2index:
                self.vectors.append(torch.FloatTensor(np.copy(self.word2vec.wv[word])).unsqueeze(0))
            else:
                self.vectors.append(torch.FloatTensor(1, self.embedding_dim).uniform_(-a, a))
        self.vectors = torch.cat(self.vectors, dim=0)
        print('shape of embedding vectors:', self.vectors.shape)

    def _generate_freq_vocab(self, corpus):
        counter = Counter()
        for line in corpus:
            counter.update(line.split(' '))
        return counter

    def collate_cnn(self, batch):
        """
        add padding for text of various lengths
        Args:
            [(text(tensor), label(tensor)), ...]
        Returns:
            tensor, tensor : text, label
        """
        text, label = zip(*batch)
        seq = [self.text2seq(t) for t in text]
        text = pad_sequence(seq, batch_first=True, padding_value=self.stoi[self.pad_token]).long()
        label = torch.FloatTensor(np.asarray(label))
        return text, label

    def text2seq(self, text):
        output = list()
        for word in text.split(' '):
            if word not in self.stoi:
                output.append(self.stoi[self.unk_token])
            else:
                output.append(self.stoi[word])
        return torch.FloatTensor(output)



