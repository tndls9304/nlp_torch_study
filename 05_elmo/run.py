import random
import torch
import math

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
    if conf_dict['use_cuda']:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    conf_dict['device'] = device
    # device = 'cpu'

    trainer = ELMOTrainer(conf_dict)

    # trainer.run()
    model = trainer.load_model(60)
    trainer.elmo = model
    test_loss_forward, test_loss_backward, test_bleu_forward, test_bleu_backward = trainer.eval_valid(60)
    print(
        f'\tTest F. Loss: {test_loss_forward:.3f} |  Test F. PPL: {math.exp(test_loss_forward):7.3f}   | Test F. BLEU: {test_bleu_forward:.3f}')
    print(
        f'\tTest B. Loss: {test_loss_backward:.3f} |  Test B. PPL: {math.exp(test_loss_backward):7.3f}   | Test B. BLEU: {test_bleu_backward:.3f}')
