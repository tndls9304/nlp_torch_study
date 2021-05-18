import torch
import json

from elmo.module.elmo import ELMO
from elmo.dataloader import ELMODataset


def json_reader(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


if __name__ == "__main__":
    config = json_reader('elmo_config.json')
    device = torch.device('cuda')

    elmo_dataset = ELMODataset(config, config['filepath'], device)
    for train_batch in elmo_dataset.train_iterator:
        print(train_batch.src)
        print(train_batch.trg)
        break

    config['output_dim'] = len(elmo_dataset.TRG.vocab)
    elmo = ELMO(config).to(device)
    output = elmo(train_batch.src)
    for o in output:
        print(o.shape)