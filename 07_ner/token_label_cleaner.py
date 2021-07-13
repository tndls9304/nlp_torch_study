import re
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class KLUENERDataset(Dataset):
    def __init__(self, text_list, bio_list):
        super().__init__()
        self.text_list = text_list
        self.bio_list = bio_list
        assert len(self.text_list) == len(self.bio_list)
        self.data_size = len(self.text_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        return self.text_list[item], self.bio_list[item]


class NERPair:
    def __init__(self, train_filepath, test_filepath, tokenizer_filepath):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.tokenizer = Tokenizer.from_file(tokenizer_filepath)
        self.train_text = list()
        self.train_bio = list()
        self.test_text = list()
        self.test_bio = list()

        self.init_dataset()

    @staticmethod
    def klue_to_text_and_bio(file_path):
        corpus_text = []
        corpus_bio = []
        with open(file_path, 'r') as f:
            _tokens = []
            _bio = []
            for cnt, line in enumerate(f.readlines()):
                if not re.search(r'^##|^\n', line):
                    token, bio = line.strip('\n').split("\t")
                    _tokens.append(token)
                    _bio.append(bio)
                elif len(_tokens) != 0:
                    assert len(_tokens) == len(_bio), "Size Mismatched"
                    corpus_text.append("".join(_tokens))
                    corpus_bio.append(_bio)
                    _tokens = []
                    _bio = []
        return corpus_text, corpus_bio

    def init_dataset(self):
        train_text, train_bio = self.klue_to_text_and_bio(self.train_filepath)
        test_text, test_bio = self.klue_to_text_and_bio(self.test_filepath)
        for text, bio in zip(train_text, train_bio):
            tmp_train_text, tmp_train_bio = self.get_token_labels(text, bio)
            self.train_text.append(tmp_train_text)
            self.train_bio.append(tmp_train_bio)
        for text, bio in zip(test_text, test_bio):
            tmp_test_text, tmp_test_bio = self.get_token_labels(text, bio)
            self.test_text.append(tmp_test_text)
            self.test_bio.append(tmp_test_bio)

    def get_token_labels(self, text: str, original_bio: list):
        cleaned_original_bio = [lbl for txt, lbl in list(zip(text, original_bio)) if txt.strip()]
        tokenized = self.tokenizer.encode(text)
        token_list = tokenized.tokens[1:-1]
        offset_list = tokenized.offsets[1:-1]
        start_index = 0
        merged_bio = []
        for offset in offset_list:
            token_length = offset[1] - offset[0]
            selected_labels = cleaned_original_bio[start_index: start_index+token_length][0]  # 가장 첫번째 bio 태그를 태그로 사용
            merged_bio.append(selected_labels)
            start_index += token_length
        assert len(token_list) == len(merged_bio), "Size Mismatched"
        return token_list, merged_bio


if __name__ == "__main__":
    train_path = 'dataset/klue-ner-v1_train.tsv'
    test_path = 'dataset/klue-ner-v1_dev.tsv'
    tokenizer_path = 'dataset/vocab.json'

    ner_pair = NERPair(train_path, test_path, tokenizer_path)
    train_dataset = KLUENERDataset(ner_pair.train_text, ner_pair.train_bio)
    test_dataset = KLUENERDataset(ner_pair.test_text, ner_pair.test_bio)
