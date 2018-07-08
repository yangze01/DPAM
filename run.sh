#!/bin/sh
tar -xvf data.tar
cat data/src_train_dev_data.txt | python make_train_dev_dataset.py "./data/one_hot_vocab_70.txt" > data/train_dev_x.txt
#exit
cat data/src_rule_data.txt | python make_rule_data.py > data/train_dev_rule.txt

python DPAM_train.py 
