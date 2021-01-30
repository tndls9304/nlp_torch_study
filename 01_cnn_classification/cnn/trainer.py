import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

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
        print('entire sentence size:', len(self.dataset))

        indices = list(range(len(self.dataset)))
        np.random.shuffle(indices)
        splits_front_indexes = [0, 1-conf_dict['VALID_RATIO']-conf_dict['TEST_RATIO'], 1-conf_dict['TEST_RATIO']]
        splits_back_indexes = [1-conf_dict['VALID_RATIO']-conf_dict['TEST_RATIO'], 1-conf_dict['TEST_RATIO'], 1]
        final_indices = list()
        for i in range(len(splits_back_indexes)):
            final_indices.append(indices[int(np.floor(len(self.dataset)*splits_front_indexes[i])):int(np.floor(len(self.dataset)*splits_back_indexes[i]))])
        train_sampler = SubsetRandomSampler(final_indices[0])
        valid_sampler = SubsetRandomSampler(final_indices[1])
        test_sampler = SubsetRandomSampler(final_indices[2])

        self.dataset.build_vocab(final_indices[0])

        conf_dict['INPUT_DIM'] = len(self.dataset.stoi)
        print('input dimension:', conf_dict['INPUT_DIM'])
        pad_idx = self.dataset.stoi[self.dataset.pad_token]
        unk_idx = self.dataset.stoi[self.dataset.unk_token]

        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=conf_dict['BATCH_SIZE'],
                                                        sampler=train_sampler, collate_fn=self.dataset.collate_cnn)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset, batch_size=conf_dict['BATCH_SIZE'],
                                                        sampler=valid_sampler, collate_fn=self.dataset.collate_cnn)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=conf_dict['BATCH_SIZE'],
                                                       sampler=test_sampler, collate_fn=self.dataset.collate_cnn)

        print('number of training data : {}'.format(len(train_sampler)))
        print('number of valid data : {}'.format(len(valid_sampler)))
        print('number of test data : {}'.format(len(test_sampler)))

        self.model = select_model(conf_dict, pad_idx, unk_idx, self.dataset.vectors, conf_dict['OPTION'])
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
            predictions = self.model(batch[0]).squeeze(1)
            loss = self.criterion(predictions, batch[1])
            acc = binary_accuracy(predictions, batch[1])
            if is_train:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self):
        return self.iterating(self.train_loader, True)

    def evaluate_valid(self):
        with torch.no_grad():
            return self.iterating(self.valid_loader, False)

    def evaluate_test(self):
        with torch.no_grad():
            return self.iterating(self.test_loader, False)

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
