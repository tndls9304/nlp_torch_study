from trainer import BERTTrainer

from general_utils.utils import json_reader


if __name__ == "__main__":
    config = json_reader('loader_config.json')
    trainer = BERTTrainer(config)
    trainer.run()
