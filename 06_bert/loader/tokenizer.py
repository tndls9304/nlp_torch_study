import codecs
import os

from konlpy.tag import Mecab

from tokenizers.implementations import BertWordPieceTokenizer


class BERTMecabTokenizer:
    def __init__(self, config):
        self.config = config
        self.corpus_dir_path = config['corpus_dir_path']
        self.mecab = Mecab()

        if not os.path.exists(os.path.join(self.config['tokenizer_path'], 'tokenizer-vocab.txt')):
            self.mecabing()
            self.training_WordPiece()
        self.tokenizer = self.load_WordPiece()

    def get_special_tokens(self):
        special_tokens = self.config['special_tokens']
        unk_tokens = ["[UNK{}]".format(i) for i in range(self.config['unk_token_num'])]
        unused_tokens = ["[UNUSED{}]".format(i) for i in range(self.config['unused_token_num'])]
        return special_tokens + unk_tokens + unused_tokens

    def _split_only_josa(self, original_sentence):
        pos_result = self.mecab.pos(original_sentence.strip())
        new_sentence = ''
        for word, pos in pos_result:
            if pos.startswith('J'):
                original_sentence = original_sentence.strip()
                new_sentence += ' ##' + word
            elif original_sentence.startswith(' '):
                original_sentence = original_sentence.strip()
                new_sentence += ' ' + word
            else:
                new_sentence += word
            original_sentence = original_sentence[len(word):]
        return new_sentence

    def mecabing(self):
        def mecab_transform_file(input_path):
            output_path = input_path[:-4] + "_mecab.txt"
            reader = codecs.open(input_path, 'r', encoding='utf-8')
            writer = codecs.open(output_path, 'w', encoding='utf-8')
            for line in reader:
                new_line = self._split_only_josa(line.strip())
                writer.write(new_line + '\n')
            print("{} file is transformed to {}".format(input_path, output_path))
            reader.close()
            writer.close()
        for file_path in os.listdir(self.corpus_dir_path):
            if file_path.endswith(".txt") and 'mecab' not in file_path:
                mecab_transform_file(os.path.join(self.corpus_dir_path, file_path))
        print('finish mecabing...')

    def training_WordPiece(self):
        tokenizer = BertWordPieceTokenizer(
            vocab=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
            wordpieces_prefix='##'
        )
        tokenizer.train([os.path.join(self.corpus_dir_path, file_path)
                         for file_path in os.listdir(self.corpus_dir_path) if 'mecab' in file_path],
                        limit_alphabet=self.config['limit_alphabet'],
                        vocab_size=self.config['vocab_size'],
                        special_tokens=self.get_special_tokens())
        print('training WordPiece is finished!')
        tokenizer.save_model(self.config['tokenizer_path'], prefix='tokenizer')
        print('tokenizer is saved in {}'.format(os.path.join(self.config['tokenizer_path'], 'tokenizer-vocab.txt')))

    def load_WordPiece(self):
        tokenizer = BertWordPieceTokenizer(
            vocab=os.path.join(self.config['tokenizer_path'], 'tokenizer-vocab.txt'),
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
            wordpieces_prefix='##'
        )
        return tokenizer
