import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import tqdm

from loader.dataloader import BERTDataset, bert_collate_fn
from loader.tokenizer import BERTMecabTokenizer
from model.bert import TrainableBERT


class BERTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
        self.config['device'] = self.device

        self.tokenizer = BERTMecabTokenizer(config)

        self.dataset = BERTDataset(config, self.tokenizer)
        self.config['input_vocab_size'] = len(self.dataset.vocab)
        self.config['pad_idx'] = self.tokenizer.tokenizer.token_to_id('[PAD]')
        print('pad index:', self.config['pad_idx'])

        self.bert = TrainableBERT(config).to(self.device)
        self.initialize_weights()

        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=bert_collate_fn)

        self.optim = AdamW(self.bert.parameters(), lr=self.config['lr'])

        self.nsp_loss = nn.NLLLoss()
        self.mlm_loss = nn.NLLLoss(ignore_index=self.config['pad_idx'])

    def train(self, epoch):
        self.iteration(epoch, self.dataloader)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_nsp_acc = 0.0
        avg_mlm_acc = 0.0
        nsp_correct = 0
        mlm_correct = 0
        nsp_element = 0
        mlm_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            nsp_output, mlm_output = self.bert.forward(data["token_embed"], data["segment_embed"])
            # print(mlm_output.shape)  (b, s, v)
            # print(data["masked_lm_label"].shape)  (b, s)
            mlm_output_accumulate = mlm_output.view(self.config['vocab_size'], -1)  # (b*s, v)
            mlm_mask_accumulate = data["masked_lm_label"].view(-1) != 0
            # print(mlm_output_accumulate.shape, mlm_mask_accumulate.shape)
            masked_mlm_output = torch.masked_select(mlm_output_accumulate, mlm_mask_accumulate).view(-1, self.config['vocab_size'])
            masked_mlm_label = torch.masked_select(data["masked_lm_label"].view(-1), mlm_mask_accumulate)
            # print(masked_mlm_output.shape)
            # print(masked_mlm_label.shape)
            nsp_losses = self.nsp_loss(nsp_output, data["is_next_sentence"])
            mlm_losses = self.mlm_loss(masked_mlm_output, masked_mlm_label)
            loss = nsp_losses + mlm_losses

            if train:
                loss.backward()

            avg_loss += loss.item()

            nsp_correct += nsp_output.argmax(dim=-1).eq(data["is_next_sentence"]).sum().item()
            nsp_element += data["is_next_sentence"].nelement()

            tmp_mlm_output = torch.masked_select(mlm_output.argmax(dim=-1), data["masked_lm_label"] != 0)
            mlm_correct += tmp_mlm_output.eq(masked_mlm_label).sum().item()
            mlm_element += tmp_mlm_output.size(0)
            # print(tmp_mlm_output, mlm_only_label)

            avg_mlm_acc += mlm_correct * 100.0 / mlm_element
            avg_nsp_acc += nsp_correct * 100.0 / nsp_element

            post_fix = {
                "epoch": epoch,
                "iter": i+1,
                "mlm_loss": "%.3f" % mlm_losses.item(),
                "nsp_loss": "%.3f" % nsp_losses.item(),
                "total_loss": "%.3f" % (avg_loss / (i + 1)),
                "avg_mlm_acc": "%.3f" % (avg_mlm_acc / (i + 1)),
                "avg_nsp_acc": "%.3f" % (avg_nsp_acc / (i + 1))
            }

            if (i+1) % self.config['log_freq'] == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=%.3f" % (epoch, str_code, avg_loss / len(data_iter)),
              "mlm_acc=%.3f" % (avg_mlm_acc / len(data_iter)),
              "nsp_acc=%.3f" % (avg_nsp_acc / len(data_iter)))

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

    def initialize_weights(self):
        for name, param in self.bert.named_parameters():
            if ("fc" in name) or ('embedding' in name):
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
            elif "layer_norm" in name:
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.constant_(param.data, 1.0)
