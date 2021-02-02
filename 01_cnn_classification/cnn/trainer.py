import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, ConcatDataset

from cnn.model.cnn import CNN1d
from cnn.dataloader import CNNWordDataset, CNNCharDataset
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
        self.conf_dict = conf_dict
        self.best_valid_loss = float('inf')
        self.best_test_loss = float('inf')
        self.train_losses = list()
        self.valid_losses = list()
        self.train_acces = list()
        self.valid_acces = list()
        self.device = device

        self.epochs = conf_dict['N_EPOCHS']
        self.save_path = conf_dict['MODEL_PATH']
        self.cv = conf_dict['CV']
        self.batch_size = conf_dict['BATCH_SIZE']
        print('current device:', device)
        self.dataset = CNNCharDataset(conf_dict, device)
        # self.dataset = CNNWordDataset(conf_dict, device)
        print('entire sentence size:', len(self.dataset))

        # indices = list(range(len(self.dataset)))
        # np.random.shuffle(indices)
        # splits_front_indexes = [0, 1-conf_dict['VALID_RATIO']-conf_dict['TEST_RATIO'], 1-conf_dict['TEST_RATIO']]
        # splits_back_indexes = [1-conf_dict['VALID_RATIO']-conf_dict['TEST_RATIO'], 1-conf_dict['TEST_RATIO'], 1]

        train_valid_size = int(np.floor(len(self.dataset) * (1-conf_dict['TEST_RATIO'])))
        train_valid_dataset, test_dataset = random_split(self.dataset, [train_valid_size, len(self.dataset) - train_valid_size])
        self.dataset.build_vocab(train_valid_dataset.indices)
        conf_dict['INPUT_DIM'] = len(self.dataset.stoi)
        print('input dimension:', conf_dict['INPUT_DIM'])
        self.pad_idx = self.dataset.stoi[self.dataset.pad_token]
        self.unk_idx = self.dataset.stoi[self.dataset.unk_token]

        valid_size = int(len(train_valid_dataset) / self.cv)
        print('train_validation_length', [valid_size for _ in range(self.cv - 1)] + [len(train_valid_dataset) - (self.cv - 1) * valid_size])
        print('sum:', len(train_valid_dataset))
        self.dataset_list = random_split(train_valid_dataset, [valid_size for _ in range(self.cv - 1)] + [len(train_valid_dataset) - (self.cv - 1) * valid_size])

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.dataset.collate_cnn)
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = self.criterion.to(device)

    def run_cv(self):
        models = list()
        optimizers = list()
        all_train_loss = list()
        all_train_acc = list()
        all_valid_loss = list()
        all_valid_acc = list()

        for i in range(self.cv):
            model, optimizer, train_losses, train_acces, valid_losses, valid_acces = self.run_single_cv(i)
            models.append(model)
            optimizers.append(optimizer)
            all_train_loss.append(np.mean(train_losses))
            all_train_acc.append(np.mean(train_acces))
            all_valid_loss.append(np.mean(valid_losses))
            all_valid_acc.append(np.mean(valid_acces))
        print(f'\tFinal Train Loss: {np.mean(all_train_loss):.3f} |Final Train Acc: {np.mean(all_train_acc) * 100:.2f}%')
        print(f'\tFinal Valid Loss: {np.mean(all_valid_loss):.3f} |Final Valid Acc: {np.mean(all_valid_acc) * 100:.2f}%')
        best_model_idx = int(np.argmax(all_valid_acc))
        best_model = models[best_model_idx]
        print(f'\tBest Valid Loss: {all_valid_loss[best_model_idx]:.3f} |Best Valid Acc: {all_valid_acc[best_model_idx] * 100:.2f}%')
        best_optimizer = optimizers[best_model_idx]

        test_loss, test_acc = self.evaluate_test(best_model, best_optimizer, self.test_loader)
        print(f'\t Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    def run_single_cv(self, i):
        train_losses = list()
        valid_losses = list()
        train_acces = list()
        valid_acces = list()
        # generate train dataset
        print(f'>>> {i + 1}th dataset is testset')
        dataset = self.dataset_list.copy()
        valid_dataset = dataset[i]
        del dataset[i]
        train_dataset = functools.reduce(lambda x, y: x + y, dataset)

        model = select_model(self.conf_dict, self.pad_idx, self.unk_idx, self.dataset.vectors, self.conf_dict['OPTION'])
        optimizer = optim.Adadelta(model.parameters(), lr=self.conf_dict['LR'])

        model = model.to(self.device)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.dataset.collate_cnn)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.dataset.collate_cnn)

        for epoch in range(self.epochs):
            # Train the model here...
            start_time = time.time()
            train_loss, train_acc = self.train(model, optimizer, train_loader)
            valid_loss, valid_acc = self.evaluate_valid(model, optimizer, valid_loader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.save_path)
                # Create prediction for val_dataset and get a score...
            print(f'CV: {i+1} Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        return model, optimizer, train_losses, train_acces, valid_losses, valid_acces

    def iterating(self, model, optimizer, iterator, is_train=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_train:
            model.train()
        else:
            model.eval()
        for text, label in iterator:
            if is_train:
                optimizer.zero_grad()
            text = text.to(self.device)
            label = label.to(self.device)
            predictions = model(text).squeeze(1)
            loss = self.criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            if is_train:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self, model, optimizer, loader):
        return self.iterating(model, optimizer, loader, True)

    def evaluate_valid(self, model, optimizer, loader):
        with torch.no_grad():
            return self.iterating(model, optimizer, loader, False)

    def evaluate_test(self, model, optimizer, loader):
        with torch.no_grad():
            return self.iterating(model, optimizer, loader, False)

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
