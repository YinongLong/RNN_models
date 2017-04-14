# -*- coding: utf-8 -*-
"""
Created on 2017/4/13 15:27

@author: YinongLong
@file: lstm.py

"""
from __future__ import print_function

import numpy as np
import char_level_lm

class LSTM(object):
    """
    实现Long Short-Term Memory
    """

    def __init__(self, input_size, output_size, cell_size,
                 learning_rate=1.0, num_steps=25):
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.num_steps = num_steps
        self.cell_state = np.zeros((cell_size, 1))
        # forget gate parameter
        self.Wfx = np.random.randn(cell_size, input_size) * 0.1
        self.Wfh = np.random.randn(cell_size, cell_size) * 0.1
        self.bf = np.zeros((cell_size, 1))
        # input gate parameter
        self.Wix = np.random.randn(cell_size, input_size) * 0.1
        self.Wih = np.random.randn(cell_size, cell_size) * 0.1
        self.bi = np.zeros((cell_size, 1))
        # candidate cell state parameter
        self.Wcx = np.random.randn(cell_size, input_size) * 0.1
        self.Wch = np.random.randn(cell_size, cell_size) * 0.1
        self.bc = np.zeros((cell_size, 1))
        # output gate parameter
        self.Wox = np.random.randn(cell_size, input_size) * 0.1
        self.Woh = np.random.randn(cell_size, cell_size) * 0.1
        self.bo = np.zeros((cell_size, 1))
        # output parameter
        self.Wy = np.random.randn(output_size, cell_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def _sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def _loss(self, inputs, targets, prev_h, prev_c):
        """
        对指定的输入输出序列，前向计算预测的损失，以及反向计算梯度
        :param inputs: list，输入序列
        :param targets: list，目标序列
        :param prev_h: array，前一次的隐藏状态
        :param prev_c: 细胞状态
        :return: 
        """
        xs, fts, its, cts, ots, hts, ats, cs = {}, {}, {}, {}, {}, {}, {}, {}
        hts[-1] = np.copy(prev_h)
        cs[-1] = np.copy(prev_c)

        # 前向传播，计算模型预测结果，以及损失值
        loss = 0.
        for t in range(self.num_steps):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1

            fts[t] = self._sigmoid(np.dot(self.Wfx, xs[t]) + np.dot(self.Wfh, hts[t-1]) + self.bf)
            its[t] = self._sigmoid(np.dot(self.Wix, xs[t]) + np.dot(self.Wih, hts[t-1]) + self.bi)
            cts[t] = np.tanh(np.dot(self.Wcx, xs[t]) + np.dot(self.Wch, hts[t-1]) + self.bc)
            cs[t] = fts[t] * cs[t-1] + its[t] * cts[t]
            ots[t] = self._sigmoid(np.dot(self.Wox, xs[t]) + np.dot(self.Woh, hts[t-1]) + self.bo)
            hts[t] = ots[t] * np.tanh(cs[t])
            zt = np.dot(self.Wy, hts[t]) + self.by
            ats[t] = np.exp(zt) / np.sum(np.exp(zt))
            loss += -np.log(ats[t][targets[t], 0])

        dWfx = np.zeros_like(self.Wfx)
        dWfh = np.zeros_like(self.Wfh)
        dbf = np.zeros_like(self.bf)

        dWix = np.zeros_like(self.Wix)
        dWih = np.zeros_like(self.Wih)
        dbi = np.zeros_like(self.bi)

        dWcx = np.zeros_like(self.Wcx)
        dWch = np.zeros_like(self.Wch)
        dbc = np.zeros_like(self.bc)

        dWox = np.zeros_like(self.Wox)
        dWoh = np.zeros_like(self.Woh)
        dbo = np.zeros_like(self.bo)

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        # 后向传播，计算参数的梯度大小
        lat_f_delta = np.zeros((self.cell_size, 1))
        lat_i_delta = np.zeros((self.cell_size, 1))
        lat_o_delta = np.zeros((self.cell_size, 1))
        lat_c_delta = np.zeros((self.cell_size, 1))
        for t in reversed(range(self.num_steps)):
            target = np.zeros((self.output_size, 1))
            target[targets[t]] = 1
            delta = ats[t] - target
            # try:
            delta_ht = np.dot(self.Wy.T, delta) + \
                       np.dot(self.Wfh.T, lat_f_delta) + np.dot(self.Wih.T, lat_i_delta) + \
                       np.dot(self.Woh.T, lat_o_delta) + np.dot(self.Wch.T, lat_c_delta)
            # except Exception:
            #     print(self.Wy.T.shape, delta.shape)
            #     return
            delta_o = ots[t] * (1 - ots[t]) * np.tanh(cs[t]) * delta_ht
            delta_cell = (1 - cs[t] * cs[t]) * ots[t] * delta_ht
            delta_f = fts[t] * (1 - fts[t]) * cs[t-1] * delta_cell
            delta_i = its[t] * (1 - its[t]) * cts[t] * delta_cell
            delta_c = (1 - cts[t] * cts[t]) * its[t] * delta_cell

            dWfx += np.dot(delta_f, xs[t].T)
            dWfh += np.dot(delta_f, hts[t-1].T)
            dbf += delta_f

            dWix += np.dot(delta_i, xs[t].T)
            dWih += np.dot(delta_i, hts[t-1].T)
            dbi += delta_i

            dWcx += np.dot(delta_c, xs[t].T)
            dWch += np.dot(delta_c, hts[t-1].T)
            dbc += delta_c

            dWox += np.dot(delta_o, xs[t].T)
            dWoh += np.dot(delta_o, hts[t-1].T)
            dbo += delta_o

            dWy += np.dot(delta, hts[t].T)
            dby += delta

            lat_c_delta = delta_c
            lat_o_delta = delta_o
            lat_i_delta = delta_i
            lat_f_delta = delta_f

        # gradient clipping
        for param in [dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy,
                      dbf, dbi, dbc, dbo, dby]:
            np.clip(param, -5, 5, out=param)
        return loss, dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy, \
               dby, dbo, dbc, dbi, dbf

    def check_gradients(self, inputs, targets, prev_h, prev_c):
        """
        进行梯度的检查
        :param inputs: 
        :param targets: 
        :param prev_h: 
        :param prev_c: 
        :return: 
        """
        num_checks = 10
        delta = 1e-5
        _, dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy, \
        dby, dbo, dbc, dbi, dbf = self._loss(inputs, targets, prev_h, prev_c)
        for param, dparam, name in zip([self.Wfx, self.Wfh, self.Wix, self.Wih,
                                        self.Wcx, self.Wch, self.Wox, self.Woh,
                                        self.Wy, self.bf, self.bi, self.bc, self.bo, self.by],
                                       [dWfx, dWfh, dWix, dWih,
                                        dWcx, dWch, dWox, dWoh,
                                        dWy, dbf, dbi, dbc, dbo, dby],
                                       ['Wfx', 'Wfh', 'Wix', 'Wih',
                                        'Wcx', 'Wch', 'Wox', 'Woh',
                                        'Wy', 'bf', 'bi', 'bc', 'bo', 'by']):
            shape_param = param.shape
            shape_dparam = dparam.shape
            assert shape_param == shape_dparam, 'error in dim, %s and %s' % ('param', 'dparam')
            for i in range(num_checks):
                index = int(np.random.uniform(0, param.size))
                old_val = param.flat[index]
                param.flat[index] = old_val + delta
                lat_loss, \
                _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h, prev_c)
                param.flat[index] = old_val - delta
                pre_loss, \
                _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h, prev_c)
                param.flat[index] = old_val
                grad_analytic = dparam.flat[index]
                grad_numerical = (lat_loss - pre_loss) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_analytic + grad_numerical)
                print('%f, %f => %e' % (grad_analytic, grad_numerical, rel_error))

    def train(self, data, char_to_ix, ix_to_char):
        """
        :param data: string，待处理的文本 
        :param char_to_ix: dict，字符到索引的映射
        :param ix_to_char: dict，索引到字符的映射
        :return: 
        """
        p = 0
        prev_h = np.zeros((self.cell_size, 1))
        prev_c = np.zeros((self.cell_size, 1))
        inputs = [char_to_ix[char] for char in data[p:(p+self.num_steps)]]
        targets = [char_to_ix[char] for char in data[(p+1):(p+self.num_steps+1)]]
        self.check_gradients(inputs, targets, prev_h, prev_c)


def main():
    data_path = '/Users/Yinong/Downloads/input.txt'
    data, char_to_ix, ix_to_char = char_level_lm.load_data(data_path)
    vocab_size = len(ix_to_char)
    lstm_model = LSTM(vocab_size, vocab_size, 200)
    lstm_model.train(data, char_to_ix, ix_to_char)


if __name__ == '__main__':
    main()
