#coding=gb18030
import sys,os
import json
import jieba
from data_helper import *


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.rstrip()
        if not line:
            continue
        _id, rule = line.split('\t')
        print(' '.join(rm_tokens(jieba.cut(rule))).encode('utf8'))
