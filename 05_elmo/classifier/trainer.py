import math
import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .dataloader import CLSDataset
from .classifier import RandomEmbeddingCLS, PreTrainedW2VEmbeddingCLS

from general_utils.utils import count_parameters, initialize_weights, epoch_time, get_bleu_simple, simple_writer


def get_acc(pred, trg):
    pred = pred.to('cpu').numpy()
    trg = trg.to('cpu').numpy()
    acc = np.average(pred == trg)
    return acc


class CLSTrainer:
    def __init__(self, config, device):
        super().__init__()
        self.model_class = RandomEmbeddingCLS
        self.config = config
        self.device = device
        self.best_valid_loss = float('inf')
        # self.best_test_loss = float('inf')
        self.train_losses = list()
        self.valid_losses = list()
        self.train_bleu = list()
        self.valid_bleu = list()

        self.clip = config['pred_clip']

        self.dataset = CLSDataset(config, device)

        src_pad_idx = self.dataset.title.vocab.stoi[self.dataset.title.pad_token]
        src_unk_idx = self.dataset.title.vocab.stoi[self.dataset.title.unk_token]
        print(f'special token idx :\n'
              f'{self.dataset.title.pad_token}: {src_pad_idx}\n'
              f'{self.dataset.title.unk_token}: {src_unk_idx}\n'
              )

        self.config['input_dim'] = len(self.dataset.title.vocab)
        self.config['output_dim'] = len(self.dataset.label.vocab)
        self.config['src_pad_idx'] = src_pad_idx

        self.classifier = self.model_class(config).to(device)
        initialize_weights(self.classifier)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.config['cls_lr'])
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train_epoch(self, epoch=0, orig_path=None, pred_path=None):
        self.classifier.train()
        epoch_loss = 0
        epoch_acc = 0
        for i, batch in tqdm(enumerate(self.dataset.train_iterator)):
            src = batch.title
            trg = batch.label.squeeze()
            self.optimizer.zero_grad()

            output = self.classifier(src)
            # [batch_size, output_dim]

            # trg = trg[:, 1:].contiguous()
            # print(trg)
            # print(trg.shape)

            loss = self.criterion(output, trg)
            pred = torch.argmax(output, -1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.clip)
            self.optimizer.step()

            acc = get_acc(pred, trg)

            epoch_loss += loss.item()
            epoch_acc += acc

        return epoch_loss / len(self.dataset.train_iterator), epoch_acc / len(self.dataset.train_iterator)

    def evaluate_epoch(self, model, iterator, epoch=0, orig_path=None, pred_path=None):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(iterator)):
                src = batch.title
                trg = batch.label.squeeze()
                output = self.classifier(src)

                loss = self.criterion(output, trg)
                pred = torch.argmax(output, -1)
                acc = get_acc(pred, trg)

                epoch_loss += loss.item()
                epoch_acc += acc

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def eval_valid(self, epoch):
        return self.evaluate_epoch(self.classifier, self.dataset.valid_iterator)

    def eval_test(self, model):
        return self.evaluate_epoch(model, self.dataset.test_iterator)

    def run(self):
        for epoch in range(self.config['n_epochs']):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            valid_loss, valid_acc = self.eval_valid(epoch)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < self.best_valid_loss:
                print(f'epoch: {epoch+1} model get better valid loss {valid_loss:.3f} than {self.best_valid_loss:.3f}')
                self.best_valid_loss = valid_loss
                # torch.save(self.classifier.state_dict(), self.config['save_path'])
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}  | Accuracy: {train_acc:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}  | Accuracy: {valid_acc:.3f}')

        best_model = self.model_class(self.config)
        best_state_dict = torch.load(self.config['save_path'])
        best_model.load_state_dict(best_state_dict)
        best_model.to(self.device)

        test_loss, test_acc = self.eval_test(best_model)
        print(f'\tTest Loss: {test_loss:.3f} |  Test PPL: {math.exp(test_loss):7.3f}  | Accuracy: {test_acc:.3f}')
