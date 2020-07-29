#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

from __future__ import division

import numpy as np
from model import *
from utils import build_graph, Data, split_validation
import pickle
import argparse
import datetime
import collections
import recUtils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=4, help='the number of steps after which the learning rate decay')
parser.add_argument('--numSkips', type=int, default=2, help='number of skips')
parser.add_argument('--embeddingSize', type=int, default=128, help='Dimension of the embedding vector')
parser.add_argument('--numSampled', type=int, default=64, help='number of negative examples to sample')


############


############
 #batch_size = 128
  #embedding_size = 128  # Dimension of the embedding vector.
 # skip_window = 1  # How many words to consider left and right.
opt = parser.parse_args()
train_data = pickle.load(open('./SR-GNN-master/datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('./SR-GNN-master/datasets/' + opt.dataset + '/test.txt', 'rb'))
#train_data[0][0] = train_data[0][0].append('1')
t = train_data[0][4] + [train_data[1][4]]
print(len(train_data[0]))


ready2GoMatrix = recUtils.serializeInputMatrix(train_data)
allSessionsLength = len(ready2GoMatrix)
averageSessionLength = int(sum([len(session) + 1 for session in train_data[0]])/len([len(session) + 1 for session in train_data[0]]))
batchSize=1
while batchSize < averageSessionLength:
    batchSize *= 2
if((batchSize - averageSessionLength) > (averageSessionLength - batchSize/2)):
    batchSize = int(batchSize/2)
data, count, item_id_map, reverse_dictionary = recUtils.build_dataset(ready2GoMatrix, allSessionsLength, )
del ready2GoMatrix
recUtils.generate_batch(batchSize, batchSize, opt.numSkips,data)
node_embeddings, reverse_dictionary = recUtils.train_graph(data, reverse_dictionary, batchSize, opt.embeddingSize, opt.numSampled, opt.numSkips,
     batchSize, averageSessionLength, './')


# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    n_node = 310
# g = build_graph(all_train_seq)
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
                 lr=opt.lr, l2=opt.l2,  step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize, lr_dc=opt.lr_dc,
                 nonhybrid=opt.nonhybrid)
print(opt)
best_result = [0, 0]
best_epoch = [0, 0]
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt, model.loss_train, model.global_step]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i, node_embeddings, item_id_map)
        _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
        loss_.append(loss)
    loss = np.mean(loss_)
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [],[]
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i, node_embeddings, item_id_map)
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
        test_loss_.append(test_loss)
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    hit = np.mean(hit)*100
    mrr = np.mean(mrr)*100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1]=epoch
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tBest Recall@20:\t%.4f\tCurrent Recall@20:\t%.4f\tBest MMR@20:\t%.4f\tCurrent MMR@20:\t%.4f\tEpoch:\t%d,\t%d'%
          (loss, test_loss, best_result[0], hit, best_result[1], mrr, best_epoch[0], best_epoch[1]))
