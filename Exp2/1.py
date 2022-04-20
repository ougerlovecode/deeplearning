#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 18:46
# @Author  :  oulinyu 
# @Site    : 
# @File    : 1.py
# @Software: PyCharm
with open("labels.txt") as f1:
    f1_data = f1.read().split('\n')
with open('test_result.txt') as f2:
    f2_data = f2.read().split('\n')
# print(f1_data[0:10]
tot = 0
for i in range(len(f1_data)):
    if f1_data[i] == f2_data[i]:
        tot += 1
print("accuracy is {}".format(tot/len(f1_data)*100))