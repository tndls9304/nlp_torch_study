import random
import torch

import numpy as np

from seq2seq.utils import json_reader
from seq2seq.trainer import S2STrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    conf_dict = json_reader('seq2seq_conf.json')
    set_seed(conf_dict['seed'])
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = S2STrainer(conf_dict, device)
    trainer.run()
