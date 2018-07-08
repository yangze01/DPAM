#coding=utf8
from __future__ import division
import numpy as np
import sys
import json
BasePath = sys.path[0]
def rm_tokens(words_list):
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words: # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list

def get_stop_words(path = BasePath + "/data/stop_word.txt"):
    file = open(path, 'rb').read().decode('utf8').split('\n')
    return set(file)

def load_laws_dict(data_path):
    with open(data_path, 'r') as f:
        data = f.read()
    laws_dict = json.loads(data)
    return laws_dict


def get_rule_data(data_path):
    rule_list = list()
    with open(data_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.rstrip().decode('utf8')
            if not line:
                continue
            rule_list.append(line)
        assert len(rule_list) == 70, "the rule number is wrong"
    return np.array(rule_list)

def get_dev_train_data(x_path, y_path):
    x_data = list()
    with open(x_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.rstrip().decode('utf8')
            if not line:
                continue
            x_data.append(line)
    y_data = np.loadtxt(y_path)
    return np.array(x_data), y_data

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    x_path = "./new_data/train_dev_x.txt"
    y_path = "./new_data/train_dev_y_labels.txt"
    x_data, y_data = get_dev_train_data(x_path, y_path)
    print(len(x_data), len(y_data))

    rule_path = "./new_data/train_rule_data.txt"
    rule_input = get_rule_data(rule_path)
    print(len(rule_input))
