
from trainer import NERTrainer

from general_utils.utils import json_reader

if __name__ == "__main__":
    config = json_reader('crf_config.json')

    trainer = NERTrainer(config)
    trainer.run()
