import math
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .dataloader import ELMODataset

from .module.elmo_simple import ELMO

from general_utils.utils import count_parameters, initialize_weights, epoch_time, get_bleu_simple, simple_writer


class ELMOTrainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.best_valid_loss = float('inf')
        self.train_losses_forward = list()
        self.train_losses_backward = list()
        self.valid_losses_forward = list()
        self.valid_losses_backward = list()
        self.train_bleu_forward = list()
        self.train_bleu_backward = list()
        self.valid_bleu_forward = list()
        self.valid_bleu_backward = list()

        self.clip = config['pred_clip']

        self.dataset = ELMODataset(config)

        src_pad_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.pad_token]
        src_unk_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.unk_token]
        src_eos_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.eos_token]
        # trg_sos_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.init_token]
        trg_pad_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.pad_token]
        trg_unk_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.unk_token]
        trg_eos_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.eos_token]
        print(f'special token idx (SRC):\n'
              f'{self.dataset.SRC.pad_token}: {src_pad_idx}\n'
              f'{self.dataset.SRC.unk_token}: {src_unk_idx}\n'
              f'{self.dataset.SRC.eos_token}: {src_eos_idx}\n'
              f'special token idx (TRG):\n'
              # f'{self.dataset.TRG.init_token}: {trg_sos_idx}\n'
              f'{self.dataset.TRG.pad_token}: {trg_pad_idx}\n'
              f'{self.dataset.TRG.unk_token}: {trg_unk_idx}\n'
              f'{self.dataset.TRG.eos_token}: {trg_eos_idx}')

        self.config['input_dim'] = len(self.dataset.SRC.vocab)
        self.config['output_dim'] = len(self.dataset.TRG.vocab)
        self.config['src_pad_idx'] = src_pad_idx
        self.config['trg_pad_idx'] = trg_pad_idx
        self.output_dim = len(self.dataset.TRG.vocab)

        self.elmo = ELMO(config).to(self.device)
        initialize_weights(self.elmo)
        print(count_parameters(self.elmo))
        self.optimizer = optim.Adam(self.elmo.parameters(), lr=self.config['lr'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx).to(self.device)

    def train_epoch(self, epoch=1, orig_path=None, pred_f_path=None, pred_b_path=None):
        self.elmo.train()
        epoch_loss_forward = 0
        epoch_loss_backward = 0
        epoch_bleu_forward = 0
        epoch_bleu_backward = 0
        epoch_target = list()
        epoch_f_pred = list()
        epoch_b_pred = list()
        for i, batch in tqdm(enumerate(self.dataset.train_iterator)):
            src = batch.src
            rsrc = batch.rsrc
            trg = batch.trg
            rtrg = batch.rtrg
            self.optimizer.zero_grad()

            forward_encoder, backward_encoder = self.elmo((src, rsrc))
            # forward_output = [batch size, trg len - 1, output_dim]
            # backward_output = [batch_size, trg len - 1, output_dim]

            forward_accumulated = forward_encoder.reshape(-1, self.output_dim)
            backward_accumulated = backward_encoder.reshape(-1, self.output_dim)

            forward_word = torch.argmax(forward_encoder, dim=-1)
            backward_word = torch.argmax(backward_encoder, dim=-1)
            bleu_forward = get_bleu_simple(forward_word, trg[:, :-1])
            bleu_backward = get_bleu_simple(backward_word, rtrg[:, :-1])
            # bleu = (bleu_forward + bleu_backward) / 2

            trg_accumulated = trg[:, :-1].contiguous().reshape(-1)  # trg = [(trg len - 1) * batch size]
            rtrg_accumulated = rtrg[:, :-1].contiguous().reshape(-1)  # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            # output = output[:, 1:]
            # output = [(trg len - 1) * batch size, output dim]
            loss_forward = self.criterion(forward_accumulated, trg_accumulated)
            loss_backward = self.criterion(backward_accumulated, rtrg_accumulated)
            loss = (loss_forward + loss_backward) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.elmo.parameters(), self.clip)
            self.optimizer.step()

            epoch_loss_forward += loss_forward.item()
            epoch_loss_backward += loss_backward.item()
            epoch_bleu_forward += bleu_forward
            epoch_bleu_backward += bleu_backward
            batch_orig_sent = self.idx2sent(trg)
            batch_f_pred_sent = self.idx2sent(forward_word)
            batch_b_pred_sent = self.idx2sent(backward_word, reverse=True)
            epoch_target += batch_orig_sent
            epoch_f_pred += batch_f_pred_sent
            epoch_b_pred += batch_b_pred_sent

        if orig_path is not None and pred_f_path is not None and pred_b_path is not None:
            if epoch == 1:
                simple_writer(orig_path, epoch_target)
            simple_writer(pred_f_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_f_pred)
            simple_writer(pred_b_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_b_pred)
        return epoch_loss_forward / len(self.dataset.train_iterator), epoch_loss_backward / len(self.dataset.train_iterator),\
               epoch_bleu_forward / len(self.dataset.train_iterator), epoch_bleu_backward / len(self.dataset.train_iterator)

    def evaluate_epoch(self, model, iterator, orig_path=None, pred_f_path=None, pred_b_path=None, epoch=1):
        model.eval()
        epoch_loss_forward = 0
        epoch_loss_backward = 0
        epoch_bleu_forward = 0
        epoch_bleu_backward = 0
        epoch_target = list()
        epoch_f_pred = list()
        epoch_b_pred = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(iterator)):
                src = batch.src
                rsrc = batch.rsrc
                trg = batch.trg
                rtrg = batch.rtrg

                forward_encoder, backward_encoder = model((src, rsrc))

                forward_accumulated = forward_encoder.reshape(-1, self.config['output_dim'])
                backward_accumulated = backward_encoder.reshape(-1, self.config['output_dim'])

                forward_word = torch.argmax(forward_encoder, dim=-1)
                backward_word = torch.argmax(backward_encoder, dim=-1)
                bleu_forward = get_bleu_simple(forward_word, trg[:, :-1])
                bleu_backward = get_bleu_simple(backward_word, rtrg[:, :-1])
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                trg_accumulated = trg[:, :-1].contiguous().reshape(-1)  # trg = [(trg len - 1) * batch size]
                rtrg_accumulated = rtrg[:, :-1].contiguous().reshape(-1)
                # trg_rev_accumulated = trg_rev.reshape(-1)
                loss_forward = self.criterion(forward_accumulated, trg_accumulated)
                loss_backward = self.criterion(backward_accumulated, rtrg_accumulated)
                # loss = (loss_forward + loss_backward) / 2

                epoch_loss_forward += loss_forward.item()
                epoch_loss_backward += loss_backward.item()
                epoch_bleu_forward += bleu_forward
                epoch_bleu_backward += bleu_backward
                batch_orig_sent = self.idx2sent(trg)
                batch_f_pred_sent = self.idx2sent(forward_word)
                batch_b_pred_sent = self.idx2sent(backward_word, reverse=True)
                epoch_target += batch_orig_sent
                epoch_f_pred += batch_f_pred_sent
                epoch_b_pred += batch_b_pred_sent

        if orig_path is not None and pred_f_path is not None and pred_b_path is not None:
            if epoch == 1:
                simple_writer(orig_path, epoch_target)
            simple_writer(pred_f_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_f_pred)
            simple_writer(pred_b_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_b_pred)
        return epoch_loss_forward / len(iterator), epoch_loss_backward / len(iterator),\
               epoch_bleu_forward / len(iterator), epoch_bleu_backward / len(iterator)

    def eval_valid(self, epoch):
        return self.evaluate_epoch(self.elmo, self.dataset.valid_iterator,
                                   self.config['valid_sentence_output'],
                                   self.config['valid_forward_pred_sentence_output'],
                                   self.config['valid_backward_pred_sentence_output'],
                                   epoch=epoch)

    def eval_test(self, model):
        return self.evaluate_epoch(model, self.dataset.test_iterator,
                                   self.config['test_sentence_output'],
                                   self.config['test_forward_pred_sentence_output'],
                                   self.config['test_backward_pred_sentence_output'])

    def idx2sent(self, index_tensor, reverse=False):  # index_tensor: [batch_size, trg_len]
        index_tensor_cpu = index_tensor.to('cpu').tolist()
        sentences = list()
        for sent_idx in index_tensor_cpu:
            if reverse:
                sent = ' '.join([self.dataset.TRG.vocab.itos[i] for i in sent_idx][::-1])
            else:
                sent = ' '.join([self.dataset.TRG.vocab.itos[i] for i in sent_idx])
            sentences.append(sent.strip())
        return sentences

    def load_model(self, epoch):
        model = ELMO(self.config)
        best_state_dict = torch.load(self.config['save_path'].format(epoch))
        print('load model with {}'.format(epoch))
        model.load_state_dict(best_state_dict)
        model.to(self.device)
        return model

    def run(self):
        best_epoch = 0
        for epoch in range(1, self.config['n_epochs']+1):
            start_time = time.time()
            train_loss_forward, train_loss_backward, train_bleu_forward, train_bleu_backward = \
                self.train_epoch(epoch, self.config['train_sentence_output'],
                                 self.config['train_forward_pred_sentence_output'],
                                 self.config['train_backward_pred_sentence_output'])
            valid_loss_forward, valid_loss_backward, valid_bleu_forward, valid_bleu_backward = self.eval_valid(epoch)

            self.train_losses_forward.append(train_loss_forward)
            self.train_losses_backward.append(train_loss_backward)
            self.train_bleu_forward.append(train_bleu_forward)
            self.train_bleu_backward.append(train_bleu_backward)
            self.valid_losses_forward.append(valid_loss_forward)
            self.valid_losses_backward.append(valid_loss_backward)
            self.valid_bleu_forward.append(valid_bleu_forward)
            self.valid_bleu_backward.append(valid_bleu_backward)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            valid_loss = (valid_loss_forward + valid_loss_backward) / 2
            if valid_loss < self.best_valid_loss:
                print(f'epoch: {epoch} model get better valid loss {valid_loss:.3f} than {self.best_valid_loss:.3f}')
                self.best_valid_loss = valid_loss
                torch.save(self.elmo.state_dict(), self.config['save_path'].format(epoch))
                best_epoch = epoch
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain F. Loss: {train_loss_forward:.3f} | Train F. PPL: {math.exp(train_loss_forward):7.3f} | Train F. BLEU: {train_bleu_forward:.3f}')
            print(f'\tTrain B. Loss: {train_loss_backward:.3f} | Train B. PPL: {math.exp(train_loss_backward):7.3f} | Train B. BLEU: {train_bleu_backward:.3f}')
            print(f'\t Val. F. Loss: {valid_loss_forward:.3f} |  Val. F. PPL: {math.exp(valid_loss_forward):7.3f} |  Val. F. BLEU: {valid_bleu_forward:.3f}')
            print(f'\t Val. B. Loss: {valid_loss_backward:.3f} |  Val. B. PPL: {math.exp(valid_loss_backward):7.3f} |  Val. B. BLEU: {valid_bleu_backward:.3f}')

        best_model = self.load_model(best_epoch)

        test_loss_forward, test_loss_backward, test_bleu_forward, test_bleu_backward = self.eval_test(best_model)
        print(f'\tTest F. Loss: {test_loss_forward:.3f} |  Test F. PPL: {math.exp(test_loss_forward):7.3f}   | Test F. BLEU: {test_bleu_forward:.3f}')
        print(f'\tTest B. Loss: {test_loss_backward:.3f} |  Test B. PPL: {math.exp(test_loss_backward):7.3f}   | Test B. BLEU: {test_bleu_backward:.3f}')
