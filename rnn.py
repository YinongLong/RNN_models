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
import char_level_lm


class RNN(object):
    """
    实现一个普通的单层Recurrent Neural Network.
    """

    def __init__(self, input_size, output_size, hidden_size,
                 num_steps=25, learning_rate=0.1, batch_size=1):
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

    def gradient_check(self, inputs, targets, prev_h):
        """
        对梯度的计算进行数值检验，小幅度的改变参数的大小，来近似计算梯度的大小
        随机的选择参数矩阵中的一个进行上下浮动，然后计算其近似的梯度大小
        :param inputs: list，输入序列
        :param targets: list，目标序列
        :param prev_h: array，提前输出的隐含层的状态
        :return: None
        """
        num_check = 10
        delta = 1e-5
        _, dWxh, dWhh, dWhy, dbh, dby, _ = self._loss(inputs, targets, prev_h)
        for param, dparam, name in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                       [dWxh, dWhh, dWhy, dbh, dby],
                                       ['Wxh', 'Whh', 'Why', 'bh', 'by']):
            shape_param = param.shape
            shape_dparam = param.shape
            assert shape_param == shape_dparam, 'Errors dims dont match: %s and %s.' % ('param', 'dparam')
            print(name)

            for i in range(num_check):
                index = int(np.random.uniform(0, param.size))
                old_val = param.flat[index]
                param.flat[index] = old_val + delta
                lat_loss, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h)
                param.flat[index] = old_val - delta
                pre_loss, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h)
                param.flat[index] = old_val
                grad_analytic = dparam.flat[index]
                grad_numerical = (lat_loss - pre_loss) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_analytic + grad_numerical)
                print('%f, %f => %e' % (grad_numerical, grad_analytic, rel_error))

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
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len_t-1]

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.output_size, 1))
        x[seed_ix] = 1
        indices = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            z = np.dot(self.Why, h) + self.by
            a = np.exp(z) / np.sum(np.exp(z))
            ix = np.random.choice(range(self.output_size), p=a.ravel())
            x = np.zeros((self.output_size, 1))
            x[ix] = 1
            indices.append(ix)
        return indices

    def train(self, data, char_to_ix, ix_to_char):
        """
        :param data: string, 用来进行建模的文本 
        :param char_to_ix: dict, 字符到索引的映射
        :param ix_to_char: dict, 索引到字符的映射
        :return: 
        """
        len_data = len(data)

        # 使用Adagrad优化算法进行优化
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)

        smooth_loss = -np.log(1.0/self.output_size) * self.num_steps

        need_check = False
        n, p = 0, 0
        # 开始模型的训练
        while True:
            # 处理完一次以后，将隐藏层状态清零
            if (p + self.num_steps + 1) >= len_data or n == 0:
                prev_h = np.zeros((self.hidden_size, 1))
                p = 0
            # for i in range(self.batch_size):
            inputs = [char_to_ix[char] for char in data[p:(p+self.num_steps)]]
            targets = [char_to_ix[char] for char in data[(p+1):(p+1+self.num_steps)]]

            if need_check:
                self.gradient_check(inputs, targets, prev_h)

            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.sample(prev_h, inputs[0], 300)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print('----------\n%s\n----------' % txt)

            loss, dWxh, dWhh, dWhy, dbh, dby, prev_h = self._loss(inputs, targets, prev_h)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, smooth_loss))

            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            p += 1
            n += 1


def main():
    data_path = '/Users/Yinong/Downloads/input.txt'
    data, char_to_ix, ix_to_char = char_level_lm.load_data(data_path)
    vocab_size = len(ix_to_char)
    rnn_model = RNN(vocab_size, vocab_size, 200)
    rnn_model.train(data, char_to_ix, ix_to_char)


if __name__ == '__main__':
    main()
