import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm

from loader.dataloader import BERTDataset, bert_collate_fn
from loader.tokenizer import BERTMecabTokenizer
from model.bert import BERT, NextSentenceClassification


class BERTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
        self.config['device'] = self.device

        self.tokenizer = BERTMecabTokenizer(config)

        self.dataset = BERTDataset(config, self.tokenizer)
        self.config['input_vocab_size'] = len(self.dataset.vocab)
        self.config['pad_idx'] = self.tokenizer.tokenizer.token_to_id('[PAD]')

        self.bert = BERT(config)
        self.nsp = NextSentenceClassification(self.config['bert_hidden_size'])

        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=bert_collate_fn)

        self.optim = Adam(self.bert.parameters(), lr=self.config['lr'])

        self.criterion = nn.NLLLoss(ignore_index=self.config['pad_idx'])

    def train(self, epoch):
        self.iteration(epoch, self.dataloader)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            bert_feature = self.bert.forward(data["token_embed"], data["segment_embed"], data['position_embed'])
            next_sent_output = self.nsp(bert_feature)
            loss = self.criterion(next_sent_output, data["is_next_sentence"])
            # loss = next_loss + mask_loss

            if train:
                loss.backward()

            correct = next_sent_output.argmax(dim=-1).eq(data["is_next_sentence"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next_sentence"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.config['log_freq'] == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def run(self):
        for epoch in range(self.config['n_epochs']):
            self.train(epoch)
            self.save(epoch)
