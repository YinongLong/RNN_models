# -*- coding: utf-8 -*-
"""
Created on 2017/4/13 15:27

@author: YinongLong
@file: rnn.py

普通单隐层RNN的实现

应用在字符级别的语言模型
"""
from __future__ import print_function

import numpy as np


class RNN(object):
    """
    实现一个普通的单层Recurrent Neural Network.
    """

    def __init__(self, input_size, output_size, hidden_size,
                 num_steps=25, learning_rate=1.0, batch_size=1):
        """
        初始化网络的参数
        :param input_size: int, 每一个时间步输入的维度
        :param output_size: int, 每一个时间布输出的维度
        :param hidden_size: int, 隐藏层的单元数
        :param num_steps: int, 网络展开的时间步数 
        :param learning_rate: float, 学习率
        :param batch_size: int, 每一次更新网络参数使用的序列样本个数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def gradient_check(self):
        pass

    def _loss(self, inputs, targets, prev_h):
        """
        :param inputs: list, 输入序列 
        :param targets: list, 目标序列
        :param prev_h: array, 隐含层的状态
        :return: 
        """
        xs, hs, zs, ys = {}, {}, {}, {}
        hs[-1] = np.copy(prev_h)
        len_t = len(inputs)
        loss = 0.
        # forward pass
        for t in range(len_t):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            zs[t] = np.dot(self.Why, hs[t]) + self.by
            ys[t] = np.exp(zs[t]) / np.sum(np.exp(zs[t]))
            loss += -np.log(ys[t][targets[t], 0])

        # backward pass
        dWxh = np.zeros((self.hidden_size, self.input_size))
        dWhh = np.zeros((self.hidden_size, self.hidden_size))
        dWhy = np.zeros((self.output_size, self.hidden_size))
        dbh = np.zeros((self.hidden_size, 1))
        dby = np.zeros((self.output_size, 1))
        pre_error = np.zeros_like(hs[-1])
        for t in reversed(range(len_t)):
            target = np.zeros((self.output_size, 1))
            target[targets[t]] = 1
            error = ys[t] - target
            dby += error
            dWhy += np.dot(error, hs[t].T)
            delta_h = np.dot(self.Why.T, error) + np.dot(self.Whh.T, pre_error)
            error = (1 - hs[t] * hs[t]) * delta_h
            dWxh += np.dot(error, xs[t].T)
            dWhh += np.dot(error, hs[t-1].T)
            dbh += error
            pre_error = error

        # gradient clipping
        for param in [dWxh, dWhh, dWhy, dby, dbh]:
            np.clip(param, -5, 5, out=param)
        return loss, dWxh, dWhh, dWhy, dbh, dby

    def train(self, data, char_to_ix, ix_to_char):
        """
        :param data: string, 用来进行建模的文本 
        :param char_to_ix: dict, 字符到索引的映射
        :param ix_to_char: dict, 索引到字符的映射
        :return: 
        """
        # 使用Adagrad优化算法进行优化
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)

        smooth_loss = -np.log(1.0/self.output_size) * self.num_steps
        pass


def main():
    pass


if __name__ == '__main__':
    main()
