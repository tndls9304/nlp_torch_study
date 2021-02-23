import torch

from cnn.trainer import CNNTrainer
from cnn.utils import json_reader

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # conf_dict = json_reader('cnn_conf.json')
    conf_dict = json_reader('cnn_char_conf.json')
    trainer = CNNTrainer(conf_dict, device)
    trainer.run_cv()

