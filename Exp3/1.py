# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
def ff(file_name,new_file_name):
    lines = open(file_name, 'r', encoding='utf-8').readlines()
    for line in lines:
        tmp = {}
        line = line.split('\t')
        tmp['head'] = line[0]
        tmp['tail'] = line[1]

        if 'placeholder' in tmp['head']:
            continue
        if 'placeholder' in tmp['tail']:
            continue
        with open(new_file_name,'a',encoding='utf-8') as f:
            f.write('\t'.join(line))


ff(file_name='data/data_train.txt',new_file_name='data/new_data_train.txt')
ff(file_name='data/data_val.txt',new_file_name='data/new_data_val.txt')