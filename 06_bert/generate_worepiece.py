from loader.tokenizer import BERTMecabTokenizer
from general_utils.utils import json_reader


if __name__ == "__main__":
    config = json_reader('loader_config.json')

    BMTokenizer = BERTMecabTokenizer(config)
