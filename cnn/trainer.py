import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from cnn.model.cnn import CNN1d
from cnn.dataloader import CNNDataset
from cnn.utils import binary_accuracy, epoch_time


def select_model(conf_dict, pad_idx, unk_idx, vectors, option=1):
    if option != 3:  # CNN-rand // W2V freeze // W2V fit
        channel_size = 1
    else:  # multichannel
        channel_size = 2
    model = CNN1d(conf_dict['INPUT_DIM'], conf_dict['EMBEDDING_DIM'], conf_dict['N_FILTERS'],
                  conf_dict['FILTER_SIZES'], conf_dict['OUTPUT_DIM'], conf_dict['DROPOUT'], pad_idx, channel_size)
    model.embedding.weight.data[unk_idx] = torch.zeros(conf_dict['EMBEDDING_DIM'])
    model.embedding.weight.data[pad_idx] = torch.zeros(conf_dict['EMBEDDING_DIM'])
    if option == 1:  # freeze
        model.embedding.weight.requires_grad = False
    if option != 0:  # W2V freeze // W2V fit // multichannel
        model.embedding.weight.data.copy_(vectors)
    else:  # CNN-rand
        model.embedding.weight.data.copy_(torch.from_numpy(np.random.rand(conf_dict['INPUT_DIM'], conf_dict['EMBEDDING_DIM'])).float())
    if option == 3:   # multichannel
        model.embedding2.weight.data[unk_idx] = torch.zeros(conf_dict['EMBEDDING_DIM'])
        model.embedding2.weight.data[pad_idx] = torch.zeros(conf_dict['EMBEDDING_DIM'])
        model.embedding2.weight.data.copy_(vectors)
        model.embedding2.weight.requires_grad = False
    return model


class CNNTrainer:
    def __init__(self, conf_dict, device):

        self.best_valid_loss = float('inf')
        self.best_test_loss = float('inf')
        self.train_losses = list()
        self.valid_losses = list()
        self.train_acces = list()
        self.valid_acces = list()

        self.dataset = CNNDataset(conf_dict, device)
        conf_dict['INPUT_DIM'] = len(self.dataset.TEXT.vocab)
        pad_idx = self.dataset.TEXT.vocab.stoi[self.dataset.TEXT.pad_token]
        unk_idx = self.dataset.TEXT.vocab.stoi[self.dataset.TEXT.unk_token]

        self.model = select_model(conf_dict, pad_idx, unk_idx, self.dataset.TEXT.vocab.vectors, conf_dict['OPTION'])
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=conf_dict['LR'])
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        self.epochs = conf_dict['N_EPOCHS']
        self.save_path = conf_dict['MODEL_PATH']

    def iterating(self, iterator, is_train=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for batch in iterator:
            if is_train:
                self.optimizer.zero_grad()
            predictions = self.model(batch.text).squeeze(1)
            loss = self.criterion(predictions, batch.label.float())
            acc = binary_accuracy(predictions, batch.label.float())
            if is_train:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self):
        return self.iterating(self.dataset.train_iterator, True)

    def evaluate_valid(self):
        return self.iterating(self.dataset.valid_iterator, False)

    def evaluate_test(self):
        return self.iterating(self.dataset.test_iterator, False)

    def run(self):
        epoch = 0
        epoch_mins = 0
        epoch_secs = 0
        train_loss = np.float('inf')
        train_acc = 0
        valid_loss = np.float('inf')
        valid_acc = 0

        for epoch in range(self.epochs):
            # Train the model here...
            start_time = time.time()
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate_valid()
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_acces.append(train_acc)
            self.valid_acces.append(valid_acc)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)
                # Create prediction for val_dataset and get a score...
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        # Create inference for test_dataset...
        test_loss, test_acc = self.evaluate_test()
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')
