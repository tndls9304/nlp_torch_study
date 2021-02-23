import codecs
import re
import time
import random
import torch
import json


def simple_reader(path):
    corpus = list()
    reader = codecs.open(path, 'r', encoding='utf-8')
    for line in reader:
        corpus.append(line.strip())
    reader.close()
    return corpus


def simple_writer(path, target):
    random.shuffle(target)
    writer = codecs.open(path, 'w', encoding='utf-8')
    for line in target:
        writer.write(line.strip() + '\n')
    writer.close()


def clean_str(string, TREC=False, lower=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'(s|ve|re|d|ll)", r" \'\1", string)
    string = re.sub(r"n\'t", r" n\'t", string)
    string = re.sub(r"([,!)(?]|\s{2,})", r" \1 ", string)
    if lower:
        return string.strip() if TREC else string.strip().lower()
    else:
        return string.strip()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_list(corpus, lower=False):
    new_corpus = list()
    for line in corpus:
        line = clean_str(line.strip(), lower=lower)
        if line.strip() == '':
            continue
        new_corpus.append(line)
    return new_corpus


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def json_reader(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data
