import time

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from module.bilstm_crf import BiLSTMCRF
from dataloader import NERDataset

from general_utils.utils import simple_writer, epoch_time


class NERTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
        self.config['device'] = self.device

        self.train_losses = list()
        self.train_acces = list()
        self.test_losses = list()
        self.test_acces = list()
        self.best_valid_loss = float('inf')

        self.dataset = NERDataset(config, self.device)
        self.sent_vocab = self.dataset.SRC.vocab
        self.tag_vocab = self.dataset.TRG.vocab
        self.config['src_vocab_size'] = len(self.sent_vocab)
        self.config['trg_vocab_size'] = len(self.tag_vocab)
        self.config['pad_idx'] = self.dataset.SRC.vocab.stoi['<pad>']
        print('pad index:', self.config['pad_idx'])

        self.model = BiLSTMCRF(self.config).to(self.device)

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, 0, 0.01)
            else:
                nn.init.constant_(param.data, 0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['lr']))

    def train_epoch(self, epoch=1, orig_path=None, pred_path=None):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_target = list()
        epoch_pred = list()
        for i, batch in tqdm(enumerate(self.dataset.train_iterator)):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()

            emit, loss, mask = self.model(src, trg)
            loss = -1 * loss
            # print(loss.shape)
            pred_trg = self.model.CRF.decode(emit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_max_norm'])
            self.optimizer.step()

            epoch_loss += loss.item()
            trg_flatten = [item for sublist in trg.to('cpu').tolist() for item in sublist]
            pred_trg_flatten = [item for sublist in pred_trg for item in sublist]
            # trg_mask = [1 if item > 3 else 0 for item in trg_flatten]
            # print(len(trg_flatten), len(pred_trg_flatten))
            # assert len(trg_flatten) == len(pred_trg_flatten)
            # trg_flatten = [item for i, item in enumerate(trg_flatten) if trg_mask[i] == 1]
            # pred_trg_flatten = [item for i, item in enumerate(pred_trg_flatten) if trg_mask[i] == 1]
            # print(len(trg_flatten), len(pred_trg_flatten))
            assert len(trg_flatten) == len(pred_trg_flatten), (trg.shape, len(pred_trg), len(pred_trg[0]), len(trg_flatten), len(pred_trg_flatten))
            epoch_acc += f1_score(trg_flatten, pred_trg_flatten, average='macro')
            batch_orig_sent = self.idx2sent(src[0], trg)
            batch_pred_sent = self.idx2sent(src[0], pred_trg)

            epoch_target += batch_orig_sent
            epoch_pred += batch_pred_sent

        if orig_path is not None and pred_path is not None:
            if epoch == 1:
                simple_writer(orig_path, epoch_target)
            simple_writer(pred_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_pred)
        return epoch_loss / len(self.dataset.train_iterator), epoch_acc / len(self.dataset.train_iterator)

    def evaluate_epoch(self, model, orig_path=None, pred_path=None, epoch=1):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        epoch_target = list()
        epoch_pred = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataset.test_iterator)):
                src = batch.src
                trg = batch.trg
                # print(src[0].shape, trg.shape)
                self.optimizer.zero_grad()

                emit, loss, mask = self.model(src, trg)
                loss = -1 * loss
                # print(loss.shape)
                pred_trg = self.model.CRF.decode(emit)

                epoch_loss += loss.item()
                trg_flatten = [item for sublist in trg.to('cpu').tolist() for item in sublist]
                pred_trg_flatten = [item for sublist in pred_trg for item in sublist]
                # trg_mask = [1 if item > 3 else 0 for item in trg_flatten]
                # print(len(trg_flatten), len(pred_trg_flatten))
                # assert len(trg_flatten) == len(pred_trg_flatten)
                # trg_flatten = [item for i, item in enumerate(trg_flatten) if trg_mask[i] == 1]
                # pred_trg_flatten = [item for i, item in enumerate(pred_trg_flatten) if trg_mask[i] == 1]
                # print(len(trg_flatten), len(pred_trg_flatten))
                assert len(trg_flatten) == len(pred_trg_flatten), (trg.shape, len(pred_trg), len(pred_trg[0]), len(trg_flatten), len(pred_trg_flatten))
                epoch_acc += f1_score(trg_flatten, pred_trg_flatten, average='macro')
                batch_orig_sent = self.idx2sent(src[0], trg)
                batch_pred_sent = self.idx2sent(src[0], pred_trg)

                epoch_target += batch_orig_sent
                epoch_pred += batch_pred_sent

            if orig_path is not None and pred_path is not None:
                if epoch == 1:
                    simple_writer(orig_path, epoch_target)
                simple_writer(pred_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_pred)

        if orig_path is not None and pred_path is not None:
            if epoch == 1:
                simple_writer(orig_path, epoch_target)
            simple_writer(pred_path.replace('.txt', f'_ep_{str(epoch).zfill(2)}.txt'), epoch_pred)
        return epoch_loss / len(self.dataset.test_iterator), epoch_acc / len(self.dataset.test_iterator)

    def idx2sent(self, sent_tensor, tag_tensor):  # index_tensor: [batch_size, trg_len]
        sent_tensor_cpu = sent_tensor.to('cpu').tolist()
        if type(tag_tensor) != list:
            tag_tensor_cpu = tag_tensor.to('cpu').tolist()
        else:
            tag_tensor_cpu = tag_tensor
        assert len(sent_tensor_cpu) == len(tag_tensor_cpu)
        sentences = list()
        for sent_idx, tag_idx in zip(sent_tensor_cpu, tag_tensor_cpu):
            sent_list = [self.sent_vocab.itos[i] for i in sent_idx]
            tag_list = [self.tag_vocab.itos[j] for j in tag_idx]
            assert len(sent_list) == len(tag_list)
            sent = ' '.join(['{}({})'.format(a, b) for a, b in zip(sent_list, tag_list)])
            sentences.append(sent.strip())
        return sentences

    def load_model(self, epoch):
        model = torch.load(self.config['save_path'].format(epoch))
        print('load model with epoch {}'.format(epoch))
        model.to(self.device)
        return model

    def run(self):
        best_epoch = 0
        for epoch in range(1, self.config['n_epochs']+1):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch, self.config['train_sentence_output'], self.config['train_pred_sentence_output'])
            test_loss, test_acc = self.evaluate_epoch(self.model, self.config['test_sentence_output'], self.config['test_pred_sentence_output'], epoch)

            self.train_losses.append(train_loss)
            self.train_acces.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_acces.append(test_acc)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if test_loss < self.best_valid_loss:
                print(f'epoch: {epoch} model get better valid loss {test_loss:.3f} than {self.best_valid_loss:.3f}')
                self.best_valid_loss = test_loss
                # torch.save(self.elmo.state_dict(), self.config['save_path'].format(epoch))
                torch.save(self.model, self.config['save_path'].format(epoch))
                best_epoch = epoch
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train F1 Acc: {train_acc:.3f}')
            print(f'\tTest Loss: {test_loss:.3f} | Test F1 Acc: {test_acc:.3f}')

        best_model = self.load_model(best_epoch)

        test_loss, test_acc = self.evaluate_epoch(best_model)
        print(f'\tBest Test Loss: {test_loss:.3f} | Best Test F1 Acc: {test_acc:.3f}')
