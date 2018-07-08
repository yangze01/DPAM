#coding=gb18030
import json
import jieba
import sys,os
import numpy as np
from data_helper import *

def get_one_hot_y_label(laws_dict, y_src):
    y_src = y_src.split(" ")
    y_ret = np.array([0]*70)
    one_hot = [laws_dict[one] for one in y_src if one in laws_dict]
    y_ret[one_hot] = 1
    return y_ret


if __name__ == "__main__":
    laws_dict_path = sys.argv[1]
    laws_dict = load_laws_dict(laws_dict_path)
    train_dev_y_array = list()
    for line in sys.stdin:
        try:
            line = line.rstrip()
            if not line:
                continue
            content, raw_labels = line.split('\t')
            cut_content = ' '.join(rm_tokens(list(jieba.cut(content))))
            y_labels = get_one_hot_y_label(laws_dict, raw_labels)
            print(cut_content.encode('utf8'))
            train_dev_y_array.append(y_labels)
        except:
            continue
    np.savetxt("./data/train_dev_y.txt", np.array(train_dev_y_array))
