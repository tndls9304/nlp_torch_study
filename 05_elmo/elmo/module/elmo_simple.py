import torch.nn as nn

from .ci_embedding import ContextIndependentEmbedding
from .stacked_lstm import StackedLSTM


class ELMO(nn.Module):
    def __init__(self, config):
        super(ELMO, self).__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.dropout = nn.Dropout(config['dropout'])
        self.layer_norm = nn.LayerNorm(config['elmo_embedding_size'])

        self.ci_embedding = ContextIndependentEmbedding(config, False)
        self.encoder_forward = StackedLSTM(config['emb_dim'], config['elmo_embedding_size'], device=self.config['device'],
                                           num_layers=config['enc_n_layers'], batch_first=True)
        self.encoder_backward = StackedLSTM(config['emb_dim'], config['elmo_embedding_size'], device=self.config['device'],
                                            num_layers=config['enc_n_layers'], batch_first=True)
        self.decoder = nn.Linear(config['elmo_embedding_size'], self.output_dim)

    def forward(self, batches):
        batch_f, batch_b = batches
        enc_forward_last, enc_backward_last, _ = self.get_encoded(batch_f, batch_b)
        enc_forward_last = enc_forward_last[-1, :, :, :].squeeze()
        enc_backward_last = enc_backward_last[-1, :, :, :].squeeze()
        # enc_forward_last = enc_output[-1, :, :, :].squeeze()
        # enc_backward_last = enc_output[-2, :, :, :].squeeze()

        # forward_pred = self.decoder(self.layer_norm(enc_forward_last))
        # backward_pred = self.decoder(self.layer_norm(enc_backward_last))
        forward_pred = self.decoder(self.layer_norm(enc_forward_last))
        backward_pred = self.decoder(self.layer_norm(enc_backward_last))
        return forward_pred, backward_pred

    def get_single_encoded(self, encoder, batch):
        batch_idx, batch_length = batch
        batch_length = batch_length.to('cpu').tolist()

        ci_embedding = self.ci_embedding(batch_idx)
        # print(ci_embedding.shape)
        # [batch, seq_len, emb_dim]
        encoder_output = encoder(self.dropout(ci_embedding), batch_length)
        return encoder_output, ci_embedding

    def get_encoded(self, batch_f, batch_b):
        encoder_output_forward, ci_embedding_f = self.get_single_encoded(self.encoder_forward, batch_f)
        encoder_output_backward, ci_embedding_b = self.get_single_encoded(self.encoder_backward, batch_b)
        # print(encoder_output.shape)
        # [num_layer, batch_size, seq_len, emb_dim*2 (forward/backward]
        # (2, 16, n, 200*2)
        # encoder_output = self.dropout(encoder_output[-1])  # select last layer
        # [batch_size, seq_len, emb_dim*2 (forward/backward]]

        # print(batch_embedding.shape)
        # [1, batch_size, seq_len, emb_dim]

        return encoder_output_forward, encoder_output_backward, ci_embedding_f

