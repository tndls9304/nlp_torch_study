import torch

from konlpy.tag import Mecab

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

        self.load_model(60)

    def load_model(self, epoch):
        self.elmo = torch.load(self.config['save_path'].format(epoch))
        print('load model with epoch {}'.format(epoch))
        self.elmo = self.elmo.to(self.device)
        self.elmo.eval()

    def get_word_embedding(self, sentence, word, order=0):
        morphed_sentence = self.tokenizer.morphs(sentence)
        assert word in morphed_sentence, 'there is no word {} in morphed sentence {}'.format(word, sentence)
        self.sampler.sentencelist2iterator([' '.join(morphed_sentence)])
        for batch in self.sampler.iterator:
            # print(batch.src[0].shape)
            elmo_rep = self.infer_model(batch)
            # print(elmo_rep.shape)
            # [batch, seq_len, emb_dim]
            for i, w in enumerate(morphed_sentence):
                if w == word:
                    if order == 0:
                        return elmo_rep[i]
                    else:
                        order -= 1

    def infer_model(self, batch):
        src = batch.src
        rsrc = batch.rsrc

        forward_output, backward_output, token_output = self.elmo.get_encoded(src, rsrc)
        # [layer, batch, seq_len, emb_dim] / [layer, batch, seq_len, emb_dim] / [batch, seq_len, emb_dim]

        elmo_representation = torch.cat([forward_output, backward_output], dim=0)
        elmo_representation = torch.mean(elmo_representation, dim=0)
        return elmo_representation.squeeze()



