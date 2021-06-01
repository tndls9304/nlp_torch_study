import torch

from elmo.module.elmo import ELMO
from konlpy.tag import Mecab
from elmo.dataloader import ELMODataset, ELMOSample

from general_utils.utils import json_reader


if __name__ == "__main__":
    config = json_reader('elmo_config.json')
    device = torch.device('cpu')
    config['device'] = device
    tokenizer = Mecab()
    elmo_dataset = ELMOSample(config)
    elmo_dataset.sent2iterator(' '.join(tokenizer.morphs('안녕? 대통령 저는 알바입니다.')))

    for train_batch in elmo_dataset.iterator:
        print(train_batch.src)
        print(train_batch.rsrc)
        break

    config['output_dim'] = len(elmo_dataset.TRG.vocab)
    elmo = ELMO(config).to(device)
    output = elmo(train_batch.src)
    for o in output:
        print(o.shape)
