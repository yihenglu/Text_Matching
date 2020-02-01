#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
参照：https://github.com/songyingxin/TextClassification-Pytorch/blob/master/run_SST.py
'''
import os
import torch
import numpy as np
from torch import nn
from torchtext import data
import torch.optim as optim

from Utils.SST2_utils import load_sst2
from Utils.utils import get_device
from train_eval import train, evaluate
# from train_eval_bak import train, evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main(config):

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print("\t \t \t the model name is {}".format(config.model_name))
    device, n_gpu = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样

    '''数据准备'''
    text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True, fix_length=config.sequence_length)
    label_field = data.LabelField(dtype=torch.long)

    train_iterator, dev_iterator, test_iterator = load_sst2(config.data_path, text_field, label_field,config.batch_size,
                                                            device, config.glove_word_file, config.cache_path)

    '''词向量准备'''
    pretrained_embeddings = text_field.vocab.vectors

    model_file = config.model_dir + 'model1.pt'

    '''模型准备'''
    if config.model_name == 'TextRNN':
        from TextRNN import TextRNN
        model = TextRNN.TextRNN(config.glove_word_dim, config.output_dim,
                                config.hidden_size, config.num_layers, config.bidirectional, config.dropout,
                                pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if config.do_train:
        train(config.epoch_num, model, train_iterator, dev_iterator, optimizer, criterion, ['0', '1'], model_file,
              config.log_dir, config.print_step, 'word')

    model.load_state_dict(torch.load(model_file))

    test_loss, test_acc, test_report = evaluate(
        model, test_iterator, criterion, ['0', '1'], 'word')
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))



if __name__ == '__main__':

    model_name = "TextRNN"  # TextRNN, TextCNN， lSTMATT, TextRCNN, TransformerText
    data_dir = "./data/"
    cache_dir = "./cache/"
    # embedding_folder = "F://NLP/NLP_resource_download/glove/glove.840B.300d/"
    embedding_folder = ""

    model_dir = "./save_models/"
    log_dir = "log/"

    if model_name == "TextCNN":
        from TextCNN import args, TextCNN

    elif model_name == "TextRNN":
        from TextRNN import args, TextRNN

    # elif model_name == "LSTMATT":
    #     from LSTM_ATT import args, LSTMATT
    #
    # elif model_name == "TextRCNN":
    #     from TextRCNN import args, TextRCNN
    #
    # elif model_name == "TransformerText":
    #     from TransformerText import args, TransformerText

    main(args.get_args(data_dir, cache_dir,
                       embedding_folder, model_dir, log_dir))