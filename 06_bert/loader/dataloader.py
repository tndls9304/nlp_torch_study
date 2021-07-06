import codecs
import os
import random
import torch

from torch.utils.data import Dataset

from general_utils.utils import simple_reader


def get_line_count(file_path_list):
    line_count_list = list()
    for file in file_path_list:
        line_count = 0
        reader = codecs.open(file, 'r', encoding='utf-8')
        for _ in reader:
            line_count += 1
        line_count -= 1
        line_count_list.append(line_count)
        reader.close()
    return line_count_list


def get_special_token_length(tokenizer, sample_length=1000):
    i = 0
    for i in range(sample_length):
        if not tokenizer.id_to_token(i).startswith('['):
            break
    return i


class BERTDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_idx = self.tokenizer.tokenizer.token_to_id('[MASK]')
        self.cls_idx = self.tokenizer.tokenizer.token_to_id('[CLS]')
        self.sep_idx = self.tokenizer.tokenizer.token_to_id('[SEP]')
        self.pad_idx = self.tokenizer.tokenizer.token_to_id('[PAD]')
        self.special_token_length = get_special_token_length(self.tokenizer.tokenizer)
        self.vocab = self.tokenizer.tokenizer.get_vocab()
        self.corpus_dir_path = config['corpus_dir_path']
        self.file_list = [os.path.join(self.corpus_dir_path, path) for path in os.listdir(self.corpus_dir_path) if not path.startswith('.') and path.endswith('txt') and 'mecab' in path]
        print('target files: ', self.file_list)
        self.file_idx = 0
        self.line_count_list = get_line_count(self.file_list)
        self.line_count = sum(self.line_count_list)
        self.tmp_line_idx = 0
        self.corpus = list()
        self.now_corpus_len = 0
        self.update_corpus()

    def get_random_token_id(self):
        return random.randint(self.special_token_length, len(self.vocab))

    def update_corpus(self):
        self.corpus = simple_reader(self.file_list[self.file_idx])
        # sorted(self.corpus, key=lambda x: len(x), reverse=True)
        self.now_corpus_len = len(self.corpus)

    def __len__(self):
        return self.line_count-1

    def __getitem__(self, item):
        encoded, is_next_sentence = self.get_sentences()
        encoded_idx, mlm_label = self.randomize_words(encoded)
        segment_idx = self.get_segment(encoded_idx)
        return encoded_idx, segment_idx, is_next_sentence, mlm_label

    def get_sentences(self):
        first_sent = self.corpus[self.tmp_line_idx]
        second_sent = self.corpus[self.tmp_line_idx+1]
        is_next_sentence = 1
        self.tmp_line_idx += 1
        if self.tmp_line_idx+1 == self.now_corpus_len-1:
            self.file_idx += 1
            self.file_idx = min(len(self.file_list)-1, self.file_idx)
            self.update_corpus()
            self.tmp_line_idx = 0

        if random.random() < self.config['random_sentence_prob']:
            indexes = list(range(self.line_count_list[self.file_idx]))
            indexes.remove(self.tmp_line_idx)
            indexes.remove(self.tmp_line_idx+1)
            random_idx = random.choice(indexes)
            second_sent = self.corpus[random_idx]
            is_next_sentence = 0

        return self.tokenizer.tokenizer.encode(first_sent, second_sent), is_next_sentence

    def masking_sentence(self, word_index_list):
        sent_length = len(word_index_list)
        idx_list = list(range(sent_length))
        random.shuffle(idx_list)
        randomize_count = max(int(sent_length * self.config['target_ratio']), 2)
        masking_count = max(int(randomize_count * self.config['mask_ratio']), 1)
        replace_count = max(int(randomize_count * self.config['random_replace_ratio']), 1)
        # remain_count = max(int(randomize_count * self.config['remain_ratio']), 1)

        masking_indexes = idx_list[0:masking_count]
        replace_indexes = idx_list[masking_count:masking_count+replace_count]
        # remain_indexes = idx_list[masking_count+replace_count:masking_count+replace_count+remain_count]
        target_indexes = idx_list[0:randomize_count]

        masked_index_list = [self.mask_idx if i in masking_indexes else self.get_random_token_id() if i in replace_indexes else word for i, word in enumerate(word_index_list)]
        masked_label = [word_idx if i in target_indexes else self.pad_idx for i, word_idx in enumerate(word_index_list)]

        return masked_index_list, masked_label

    def balancing_sentence_length(self, first_sent, second_sent):
        if len(first_sent) + len(second_sent) > self.config['max_len'] - 3:
            first_sent = first_sent[0:int((self.config['max_len']-3)/2)]
            second_sent = second_sent[0:int((self.config['max_len']-3)-len(first_sent))]
        # if len(first_sent) + len(second_sent) > self.config['max_len'] - 3:
        # print(len(first_sent), len(second_sent))
        assert len(first_sent) + len(second_sent) <= self.config['max_len'] - 3
        return first_sent, second_sent

    def randomize_words(self, encoded):
        first_sep_idx = encoded.ids.index(self.sep_idx)
        first_sent_encoded = encoded.ids[1:first_sep_idx]
        second_sent_encoded = encoded.ids[first_sep_idx+1:-1]

        first_sent_encoded, second_sent_encoded = self.balancing_sentence_length(first_sent_encoded, second_sent_encoded)

        first_sent_encoded, first_mlm_label = self.masking_sentence(first_sent_encoded)
        second_sent_encoded, second_mlm_label = self.masking_sentence(second_sent_encoded)

        output = [self.cls_idx] + first_sent_encoded + [self.sep_idx] + second_sent_encoded + [self.sep_idx]
        mlm_label = [self.pad_idx] + first_mlm_label + [self.pad_idx] + second_mlm_label + [self.pad_idx]

        assert len(output) == len(mlm_label)
        assert len(output) <= self.config['max_len']

        return output, mlm_label

    def get_segment(self, encoded):
        sentence_length = len(encoded)
        first_sep_idx = encoded.index(self.sep_idx)
        output = [1] * (first_sep_idx + 1) + [2] * (sentence_length - first_sep_idx - 1)
        return output


def bert_collate_fn(batch):
    def padding(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.tensor(seq[:end]).long()
        return padded_seqs, lengths

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    token_embedding, segment_embedding, nsp_label, mlm_label = zip(*batch)

    batch_dict = dict()
    batch_dict['token_embed'], _ = padding(token_embedding)
    batch_dict['segment_embed'], _ = padding(segment_embedding)
    batch_dict['masked_lm_label'], _ = padding(mlm_label)
    batch_dict['is_next_sentence'] = torch.Tensor(nsp_label).long()
    return batch_dict
