# -*- coding: utf-8 -*-
"""
Created on 2017/4/13 15:31

@author: YinongLong
@file: char_level_lm.py

"""
from __future__ import print_function

import os


def load_data(data_path):
    data = open(data_path).read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return data, char_to_ix, ix_to_char


def main():
    data_dir = 'C:/Users/YinongLong/Desktop'
    data_path = os.path.join(data_dir, 'input.txt')
    load_data(data_path)


if __name__ == '__main__':
    main()