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
                 learning_rate=0.5, batch_size=128, epoch=10, num_steps=40):
        """
        初始化整个LSTM模型
        :param input_size: int，输入单元的维数 
        :param output_size: int，输出单元的维数
        :param cell_size: int，cell state的维数
        :param learning_rate: float，学习率
        :param num_steps: int，模型展开的时间步数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate

        # forget gate parameter initialization
        self.Wfx = np.random.randn(cell_size, input_size) * 0.1
        self.Wfh = np.random.randn(cell_size, cell_size) * 0.1
        self.bf = np.zeros((cell_size, 1))

        # input gate parameter initialization
        self.Wix = np.random.randn(cell_size, input_size) * 0.1
        self.Wih = np.random.randn(cell_size, cell_size) * 0.1
        self.bi = np.zeros((cell_size, 1))

        # candidate cell state parameter initialization
        self.Wcx = np.random.randn(cell_size, input_size) * 0.1
        self.Wch = np.random.randn(cell_size, cell_size) * 0.1
        self.bc = np.zeros((cell_size, 1))

        # output gate parameter initialization
        self.Wox = np.random.randn(cell_size, input_size) * 0.1
        self.Woh = np.random.randn(cell_size, cell_size) * 0.1
        self.bo = np.zeros((cell_size, 1))

        # output parameter initialization
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
        xts, fts, its, cts, ots, hts, ats, cell_states = {}, {}, {}, {}, {}, {}, {}, {}

        hts[-1] = np.copy(prev_h)
        cell_states[-1] = np.copy(prev_c)

        # 前向传播，计算模型预测结果，以及损失值
        loss = 0.
        for t in range(self.num_steps):
            xts[t] = np.zeros((self.input_size, 1))
            xts[t][inputs[t]] = 1

            fts[t] = self._sigmoid(np.dot(self.Wfx, xts[t]) + np.dot(self.Wfh, hts[t-1]) + self.bf)
            its[t] = self._sigmoid(np.dot(self.Wix, xts[t]) + np.dot(self.Wih, hts[t-1]) + self.bi)
            cts[t] = np.tanh(np.dot(self.Wcx, xts[t]) + np.dot(self.Wch, hts[t-1]) + self.bc)
            cell_states[t] = fts[t] * cell_states[t-1] + its[t] * cts[t]
            ots[t] = self._sigmoid(np.dot(self.Wox, xts[t]) + np.dot(self.Woh, hts[t-1]) + self.bo)
            hts[t] = ots[t] * np.tanh(cell_states[t])
            zt = np.dot(self.Wy, hts[t]) + self.by
            ats[t] = np.exp(zt) / np.sum(np.exp(zt))
            try:
                loss += -np.log(ats[t][targets[t], 0])
            except Exception:
                print(ats[t].shape, targets, t)


        fts[self.num_steps] = np.zeros((self.cell_size, 1))

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
        lat_cell_delta = np.zeros((self.cell_size, 1))
        for t in reversed(range(self.num_steps)):
            target = np.zeros((self.output_size, 1))
            target[targets[t]] = 1
            delta = ats[t] - target
            delta_ht = np.dot(self.Wy.T, delta) + \
                       np.dot(self.Wfh.T, lat_f_delta) + np.dot(self.Wih.T, lat_i_delta) + \
                       np.dot(self.Woh.T, lat_o_delta) + np.dot(self.Wch.T, lat_c_delta)
            delta_o = ots[t] * (1 - ots[t]) * np.tanh(cell_states[t]) * delta_ht
            delta_cell = (1 - cell_states[t] * cell_states[t]) * ots[t] * delta_ht + lat_cell_delta * fts[t+1]
            delta_f = fts[t] * (1 - fts[t]) * cell_states[t-1] * delta_cell
            delta_i = its[t] * (1 - its[t]) * cts[t] * delta_cell
            delta_c = (1 - cts[t] * cts[t]) * its[t] * delta_cell

            dWfx += np.dot(delta_f, xts[t].T)
            dWfh += np.dot(delta_f, hts[t-1].T)
            dbf += delta_f

            dWix += np.dot(delta_i, xts[t].T)
            dWih += np.dot(delta_i, hts[t-1].T)
            dbi += delta_i

            dWcx += np.dot(delta_c, xts[t].T)
            dWch += np.dot(delta_c, hts[t-1].T)
            dbc += delta_c

            dWox += np.dot(delta_o, xts[t].T)
            dWoh += np.dot(delta_o, hts[t-1].T)
            dbo += delta_o

            dWy += np.dot(delta, hts[t].T)
            dby += delta

            lat_c_delta = delta_c
            lat_o_delta = delta_o
            lat_i_delta = delta_i
            lat_f_delta = delta_f
            lat_cell_delta = delta_cell

        # gradient clipping
        for param in [dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy,
                      dbf, dbi, dbc, dbo, dby]:
            np.clip(param, -5, 5, out=param)
        return loss, dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy, \
               dby, dbo, dbc, dbi, dbf, hts[self.num_steps-1], cell_states[self.num_steps-1]

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
        dby, dbo, dbc, dbi, dbf, _, _ = self._loss(inputs, targets, prev_h, prev_c)
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
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h, prev_c)
                param.flat[index] = old_val - delta
                pre_loss, \
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self._loss(inputs, targets, prev_h, prev_c)
                param.flat[index] = old_val
                grad_analytic = dparam.flat[index]
                grad_numerical = (lat_loss - pre_loss) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_analytic + grad_numerical)
                print('%f, %f => %e %s' % (grad_analytic, grad_numerical, rel_error, name))

    def sample(self, h, cell_state, seed_ix, n=200):
        x = np.zeros((self.input_size, 1))
        x[seed_ix] = 1
        indices = list()
        indices.append(seed_ix)
        for t in range(n):
            ft = self._sigmoid(np.dot(self.Wfx, x) + np.dot(self.Wfh, h) + self.bf)
            it = self._sigmoid(np.dot(self.Wix, x) + np.dot(self.Wih, h) + self.bi)
            ct = np.tanh(np.dot(self.Wcx, x) + np.dot(self.Wch, h) + self.bc)
            cell_state = cell_state * ft + it * ct
            ot = self._sigmoid(np.dot(self.Wox, x) + np.dot(self.Woh, h) + self.bo)
            h = ot * np.tanh(cell_state)
            zt = np.dot(self.Wy, h) + self.by
            at = np.exp(zt) / np.sum(np.exp(zt))
            ix = np.random.choice(range(self.output_size), p=at.ravel())
            x = np.zeros((self.input_size, 1))
            x[ix] = 1
            indices.append(ix)
        return indices

    def train(self, data, char_to_ix, ix_to_char):
        """
        :param data: string，待处理的文本 
        :param char_to_ix: dict，字符到索引的映射
        :param ix_to_char: dict，索引到字符的映射
        :return: 
        """
        len_data = len(data)
        mWfx, mWfh = np.zeros_like(self.Wfx), np.zeros_like(self.Wfh)
        mWix, mWih = np.zeros_like(self.Wix), np.zeros_like(self.Wih)
        mWcx, mWch = np.zeros_like(self.Wcx), np.zeros_like(self.Wch)
        mWox, mWoh = np.zeros_like(self.Wox), np.zeros_like(self.Woh)
        mWy = np.zeros_like(self.Wy)
        mby, mbo, mbc = np.zeros_like(self.by), np.zeros_like(self.bo), np.zeros_like(self.bc)
        mbi, mbf = np.zeros_like(self.bi), np.zeros_like(self.bf)
        smooth_loss = -np.log(1.0 / self.output_size) * self.num_steps
        for i in range(self.epoch):
            prev_h = np.zeros((self.cell_size, 1))
            prev_c = np.zeros((self.cell_size, 1))
            p, n = 0, 0
            while (p + 1 + self.num_steps) < len_data:
                inputs = [char_to_ix[char] for char in data[p:(p+self.num_steps)]]
                targets = [char_to_ix[char] for char in data[(p+1):(p+self.num_steps+1)]]
                if n % 100 == 0:
                    sample_ix = self.sample(prev_h, prev_c, inputs[0])
                    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                    print('----------\n%s\n----------' % txt)

                loss, dWfx, dWfh, dWix, dWih, dWcx, dWch, dWox, dWoh, dWy,\
                dby, dbo, dbc, dbi, dbf, prev_h, prev_c = self._loss(inputs, targets, prev_h, prev_c)
                smooth_loss = smooth_loss * 0.999 + 0.001 * loss
                if n % 100 == 0:
                    print('iter %d, loss %f' % (n, smooth_loss))

                for param, dparam, mem in zip([self.Wfx, self.Wfh, self.bf,
                                               self.Wix, self.Wih, self.bi,
                                               self.Wcx, self.Wch, self.bc,
                                               self.Wox, self.Woh, self.bo,
                                               self.Wy, self.by],
                                              [dWfx, dWfh, dbf,
                                               dWix, dWih, dbi,
                                               dWcx, dWch, dbc,
                                               dWox, dWoh, dbo,
                                               dWy, dby],
                                              [mWfx, mWfh, mbf,
                                               mWix, mWih, mbi,
                                               mWcx, mWch, mbc,
                                               mWox, mWoh, mbo,
                                               mWy, mby]):
                    mem += dparam * dparam
                    param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
                p += 1
                n += 1


def main():
    data_path = '/Users/Yinong/Downloads/input.txt'
    data, char_to_ix, ix_to_char = char_level_lm.load_data(data_path)
    vocab_size = len(ix_to_char)
    lstm_model = LSTM(vocab_size, vocab_size, 200)
    lstm_model.train(data, char_to_ix, ix_to_char)


if __name__ == '__main__':
    main()
