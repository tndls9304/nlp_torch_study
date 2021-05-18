import math
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .dataloader import ELMODataset

from .module.elmo import ELMO

from general_utils.utils import count_parameters, initialize_weights, epoch_time, get_bleu_simple, simple_writer


class ELMOTrainer:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.best_valid_loss = float('inf')
        # self.best_test_loss = float('inf')
        self.train_losses = list()
        self.valid_losses = list()
        self.train_bleu = list()
        self.valid_bleu = list()

        self.clip = config['pred_clip']

        self.dataset = ELMODataset(config, config['filepath'], device)

        src_pad_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.pad_token]
        src_unk_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.unk_token]
        src_eos_idx = self.dataset.SRC.vocab.stoi[self.dataset.SRC.eos_token]
        trg_sos_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.init_token]
        trg_pad_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.pad_token]
        trg_unk_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.unk_token]
        trg_eos_idx = self.dataset.TRG.vocab.stoi[self.dataset.TRG.eos_token]
        print(f'special token idx (SRC):\n'
              f'{self.dataset.SRC.pad_token}: {src_pad_idx}\n'
              f'{self.dataset.SRC.unk_token}: {src_unk_idx}\n'
              f'{self.dataset.SRC.eos_token}: {src_eos_idx}\n'
              f'special token idx (TRG):\n'
              f'{self.dataset.TRG.init_token}: {trg_sos_idx}\n'
              f'{self.dataset.TRG.pad_token}: {trg_pad_idx}\n'
              f'{self.dataset.TRG.unk_token}: {trg_unk_idx}\n'
              f'{self.dataset.TRG.eos_token}: {trg_eos_idx}')

        self.config['input_dim'] = len(self.dataset.SRC.vocab)
        self.config['output_dim'] = len(self.dataset.TRG.vocab)
        self.config['src_pad_idx'] = src_pad_idx
        self.config['trg_pad_idx'] = trg_pad_idx

        self.elmo = ELMO(config).to(self.device)
        initialize_weights(self.elmo)
        self.optimizer = optim.Adam(self.elmo.parameters(), lr=self.config['lr'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx).to(device)

    def train_epoch(self, epoch=0, orig_path=None, pred_path=None):
        self.elmo.train()
        epoch_loss = 0
        epoch_bleu = 0
        epoch_target = list()
        epoch_pred = list()
        for i, batch in enumerate(self.dataset.train_iterator):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()
            output = self.elmo(src)
            # output = [batch size, trg len - 1, output dim]

            output_word = torch.argmax(output, dim=-1)
            trg = trg[:, 1:]
            bleu = get_bleu_simple(output_word, trg)
            output_dim = output.shape[-1]

            output_accumulated = output.view(-1, output_dim)
            # output = [(trg len - 1) * batch size, output dim]
            trg_accumulated = trg.contiguous().view(-1)
            # output = output[:, 1:]
            # output = [(trg len - 1) * batch size, output dim]

            # trg_accumulated = trg.reshape(-1)  # trg = [(trg len - 1) * batch size]
            # print(trg.shape)
            loss = self.criterion(output_accumulated, trg_accumulated)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.elmo.parameters(), self.clip)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_bleu += bleu
            batch_orig_sent = self.idx2sent(trg)
            batch_pred_sent = self.idx2sent(output_word)
            epoch_target += batch_orig_sent
            epoch_pred += batch_pred_sent

        if orig_path is not None and pred_path is not None:
            simple_writer(orig_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_target)
            simple_writer(pred_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_pred)
        return epoch_loss / len(self.dataset.train_iterator), epoch_bleu / len(self.dataset.train_iterator)

    def evaluate_epoch(self, model, iterator, epoch=0, orig_path=None, pred_path=None):
        model.eval()
        epoch_loss = 0
        epoch_bleu = 0
        epoch_target = list()
        epoch_pred = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(iterator)):
                src = batch.src
                trg = batch.trg

                output, _ = model(src)
                output_accumulated = output.reshape(-1, self.config['output_dim'])  # output = [(trg len - 1) * batch size, output dim]
                output_word = torch.argmax(output, dim=-1)

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                trg = trg[:, 1:].contiguous()
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                trg_accumulated = trg.reshape(-1)  # trg = [(trg len - 1) * batch size]

                loss = self.criterion(output_accumulated, trg_accumulated)
                bleu = get_bleu_simple(output_word, trg)

                epoch_loss += loss.item()
                epoch_bleu += bleu
                batch_orig_sent = self.idx2sent(trg)
                batch_pred_sent = self.idx2sent(output_word)
                epoch_target += batch_orig_sent
                epoch_pred += batch_pred_sent

        if orig_path is not None and pred_path is not None:
            simple_writer(orig_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_target)
            simple_writer(pred_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_pred)
        return epoch_loss / len(iterator), epoch_bleu / len(iterator)

    def eval_valid(self, epoch):
        return self.evaluate_epoch(self.elmo, self.dataset.valid_iterator, epoch, self.config['valid_sentence_output'], self.config['valid_pred_sentence_output'])

    def eval_test(self, model):
        return self.evaluate_epoch(model, self.dataset.test_iterator, orig_path=self.config['test_sentence_output'], pred_path=self.config['test_pred_sentence_output'])

    def idx2sent(self, index_tensor):  # index_tensor: [batch_size, trg_len]
        index_tensor_cpu = index_tensor.to('cpu').tolist()
        sentences = list()
        for sent_idx in index_tensor_cpu:
            sent = ' '.join([self.dataset.TRG.vocab.itos[i] for i in sent_idx])
            sentences.append(sent.strip())
        return sentences

    def run(self):
        for epoch in range(self.config['n_epochs']):
            start_time = time.time()
            train_loss, train_bleu = self.train_epoch(epoch, self.config['train_sentence_output'], self.config['train_pred_sentence_output'])
            valid_loss, valid_bleu = self.eval_valid(epoch)
            self.train_losses.append(train_loss)
            self.train_bleu.append(train_bleu)
            self.valid_losses.append(valid_loss)
            self.valid_bleu.append(valid_bleu)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < self.best_valid_loss:
                print(f'epoch: {epoch+1} model get better valid loss {valid_loss:.3f} than {self.best_valid_loss:.3f}')
                self.best_valid_loss = valid_loss
                torch.save(self.elmo.state_dict(), self.config['save_path'])
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train BLEU: {train_bleu:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. BLEU: {valid_bleu:.3f}')

        best_model = ELMO(self.config)
        best_state_dict = torch.load(self.config['save_path'])
        best_model.load_state_dict(best_state_dict)
        best_model.to(self.device)

        test_loss, test_bleu = self.eval_test(best_model)
        print(f'\tTest Loss: {test_loss:.3f} |  Test PPL: {math.exp(test_loss):7.3f}   | Test BLEU: {test_bleu:.3f}')
