import torch

from elmo.module.elmo import ELMO
from elmo.dataloader import ELMODataset

from general_utils.utils import json_reader


if __name__ == "__main__":
    config = json_reader('elmo_config.json')
    device = torch.device('cpu')
    config['device'] = device

    elmo_dataset = ELMODataset(config, build_vocab=False)

    """
        for train_batch in elmo_dataset.train_iterator:
            print(train_batch.src)
            print(train_batch.trg)
            break
    
        config['output_dim'] = len(elmo_dataset.TRG.vocab)
        elmo = ELMO(config).to(device)
        output = elmo(train_batch.src)
        for o in output:
            print(o.shape)
    """