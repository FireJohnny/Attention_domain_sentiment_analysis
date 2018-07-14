#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: data_process.py
@time: 2018/6/25 10:24
"""

import numpy as np
import re
import codecs

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[\n\t\r]","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file=None):
    # Load data from files
    x_text = []
    labels = []
    with codecs.open(positive_data_file,encoding="utf-8") as f:
        for l in f.readlines():
            _label = [1,0]

            x_text.append(clean_str(l))
            labels.append(_label)

    with codecs.open(negative_data_file,encoding="utf-8") as f:
        for l in f.readlines():
            _label = [0,1]

            x_text.append(clean_str(l))
            labels.append(_label)

    return x_text,np.array(labels)

def load_unlabel_data(unlabel_file,_label = None):
    x_text = []
    label = []
    with codecs.open(unlabel_file,encoding="utf-8") as f:
        for l in f.readlines():
            label.append(_label)
            x_text.append(clean_str(l))
    return x_text#,np.array(label)
def load_pre_data(pre_data):
    p_data = list(codecs.open(pre_data, "r",encoding="utf-8").readlines())
    data = [s.strip(" ") for s in p_data]

    return data

def creat_batch(x1_v,x2_v,x1_t,x2_t,x1_labels,x2_labels,batch_size = 64,random_data = True):
    data_len  = len(x1_v)
    num_batch_per_epoch = int((data_len-1)/batch_size)+1
    if random_data:
        shuffle_indices_x1 = np.random.permutation(np.arange(data_len))
        shuffle_indices_x2 = np.random.permutation(np.arange(data_len))
        shuffle_x1 = np.array(x1_v)[shuffle_indices_x1]
        shuffle_x2 = np.array(x2_v)[shuffle_indices_x2]
        shuffle_x1_tf = np.array(x1_t)[shuffle_indices_x1]
        shuffle_x2_tf = np.array(x2_t)[shuffle_indices_x2]

        shuffle_x1_lablels = x1_labels[shuffle_indices_x1]
        shuffle_x2_lablels = x2_labels[shuffle_indices_x2]
    for batch in range(num_batch_per_epoch):
        start_index = batch*batch_size
        end_index = min((batch+1)*batch_size,data_len)
        yield shuffle_x1[start_index:end_index],shuffle_x2[start_index:end_index],shuffle_x1_tf[start_index:end_index],\
              shuffle_x2_tf[start_index:end_index],shuffle_x1_lablels[start_index:end_index],shuffle_x2_lablels[start_index:end_index]

if __name__ == '__main__':
    pass


