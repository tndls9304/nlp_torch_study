import random
import torch

import numpy as np

from elmo.trainer import ELMOTrainer
from general_utils.utils import json_reader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    conf_dict = json_reader('elmo_config.json')
    set_seed(conf_dict['seed'])
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    trainer = ELMOTrainer(conf_dict, device)
    trainer.run()
