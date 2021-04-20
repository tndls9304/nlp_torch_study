import codecs
import json
import torch.nn as nn

from sacrebleu import corpus_bleu
from nltk.tokenize import word_tokenize


def simple_reader(path):
    corpus = list()
    reader = codecs.open(path, 'r', encoding='utf-8')
    for line in reader:
        corpus.append(line.strip())
    reader.close()
    return corpus


def simple_writer(path, target):
    # random.shuffle(target)
    writer = codecs.open(path, 'w', encoding='utf-8')
    for line in target:
        writer.write(line.strip() + '\n')
    writer.close()
    print('{} sentences are written'.format(len(target)))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def json_reader(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)
    return model


def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {param_count:,} trainable parameters')


def tokenize_fr_nltk(text):
    return word_tokenize(text, language='french')


def tokenize_en_nltk(text):
    return word_tokenize(text, language='english')


def get_bleu_simple(sys, ref):
    int2char = lambda x: ' '.join([str(xx) for xx in x])
    sys_cpu = sys.to('cpu').tolist()
    ref_cpu = ref.to('cpu').tolist()
    sys_cpu_list = [int2char(s) for s in sys_cpu]
    ref_cpu_list = [int2char(s) for s in ref_cpu]
    bleu = corpus_bleu(sys_cpu_list, [ref_cpu_list])
    return bleu.score
