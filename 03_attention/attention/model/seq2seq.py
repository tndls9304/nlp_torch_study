import random

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = config['teacher_forcing']
        self.trg_pad_idx = config['trg_pad_idx']
        self.trg_eos_idx = config['trg_eos_idx']
        assert encoder.enc_hid_dim == decoder.dec_hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src_batch, trg_batch, is_train=True):
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        src, src_length = src_batch
        trg, trg_length = trg_batch
        # print('s', src.shape)  [batch, length]
        # print('t', trg.shape)  [batch, length]
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src, src_length)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[:, 0]

        outputs = torch.zeros(1, batch_size, trg_vocab_size).to(self.device)
        search_eos = torch.zeros(batch_size, 1).to(self.device)
        t = 0
        loop = True
        # print(trg.shape)
        while loop:
            # print('di', dec_input.shape, hidden.shape)
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token
            output = output.unsqueeze(0)
            # print(outputs.shape, output.shape)
            outputs = torch.cat([outputs, output], dim=0)  # [len, batch, feature]
            # outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.squeeze().argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if t + 2 == trg.shape[1]:
                break
            t += 1
            search_eos = torch.logical_or(search_eos, top1 == self.trg_eos_idx)
            if torch.all(search_eos):
                padding = torch.zeros((trg_len - outputs.shape[0], outputs.shape[1], outputs.shape[2])).to(self.device)
                padding.fill_(self.trg_pad_idx)
                outputs = torch.cat([outputs, padding], dim=0)
                break
                # loop = False
            if is_train:
                dec_input = trg[:, t] if teacher_force else top1
            else:
                dec_input = top1
            # print(trg[:, t].shape, top1.shape, dec_input.shape)  [batch_size]
        # print(outputs.shape, trg.shape)
        return outputs.permute(1, 0, 2)
