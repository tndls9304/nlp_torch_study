import torch

from konlpy.tag import Mecab

from .module.elmo import ELMO
from .dataloader import ELMOSample
from general_utils.utils import pickle_reader


class ELMOEmbedding:
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.elmo = None
        self.sampler = ELMOSample(config)
        self.trg_vocab = pickle_reader(config['trg_field_path'])
        self.tokenizer = Mecab()

        self.config['output_dim'] = len(self.trg_vocab)
        print('set output_dim as {}'.format(self.config['output_dim']))

    def load_model(self, epoch):
        self.elmo = ELMO(self.config)
        self.elmo.load_state_dict(torch.load(self.config['save_path'].format(epoch)))
        self.elmo.to(self.device)

    def get_word_embedding(self, sentence, word):
        morphed_sentence = self.tokenizer.morphs(sentence)
        assert word in morphed_sentence, 'there is no word {} in morphed sentence {}'.format(word, sentence)
        self.sampler.sentencelist2iterator([morphed_sentence])
        # for batch in self.sampler.iterator:



