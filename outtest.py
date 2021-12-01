from utils import *

import time
import json
import copy
import argparse
import numpy as np
from collections import defaultdict, Counter

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--outtest', type=str, required=True, choices=['drugbank','ttd'])
    parser.add_argument('--model', type=str, default='best_model/final_model.pth')
    
    return parser

def outtest_predict(args):
    with open('dataset/data_split_dict.json','r') as fin:
        data_split_dict = json.load(fin)

    ### load train data
    train_nid = th.LongTensor(data_split_dict['train'])
    tar_nid = th.LongTensor(data_split_dict['tar'])

    dst_nodes_train = th.LongTensor(np.load('dataset/graph_construct/dst_nodes_train.npy'))
    src_nodes_train = th.LongTensor(np.load('dataset/graph_construct/src_nodes_train.npy'))
    edge_weights_train = th.Tensor(np.load('dataset/graph_construct/edge_weights_train.npy'))
    node_features_train = th.Tensor(np.load('dataset/graph_construct/node_features_train.npy'))

    ### load outtest data
    with open('dataset/{}_outtest/{}name2id_train_dict.json'.format(args.outtest,args.outtest),'r') as fin:
        outtest_name2id = json.load(fin)
    outtest_nid = th.LongTensor(list(sorted(outtest_name2id.values())))
    outtest_tar_nid = th.cat([outtest_nid, tar_nid])

    dst_nodes_outtest = th.LongTensor(np.load('dataset/{}_outtest/{}_dst_nodes.npy'.format(args.outtest,args.outtest)))
    src_nodes_outtest = th.LongTensor(np.load('dataset/{}_outtest/{}_src_nodes.npy'.format(args.outtest,args.outtest)))
    edge_weights_outtest = th.Tensor(np.load('dataset/{}_outtest/{}_edge_weights.npy'.format(args.outtest,args.outtest))).view(-1,1)
    node_features_outtest = th.Tensor(np.load('dataset/{}_outtest/{}_features.npy'.format(args.outtest,args.outtest)))

    ### merge train data and outtest data
    dst_nodes_all = th.cat([dst_nodes_train, dst_nodes_outtest])
    src_nodes_all = th.cat([src_nodes_train, src_nodes_outtest])
    edges_all = (src_nodes_all, dst_nodes_all)
    edge_weights_all = th.cat([edge_weights_train, edge_weights_outtest])
    node_features_all = th.cat([node_features_train, node_features_outtest])

    edge_weights_all = edge_weights_all.to(device)
    node_features_all = node_features_all.to(device)
    g_all = dgl.graph(edges_all)
    prepare_mp(g_all)

    ### load outtest data labels
    outtest_pos_labels_dst = th.LongTensor(np.load('dataset/{}_outtest/{}_pos_labels_dst.npy'.format(args.outtest,args.outtest)))
    outtest_pos_labels_src = th.LongTensor(np.load('dataset/{}_outtest/{}_pos_labels_src.npy'.format(args.outtest,args.outtest)))
    outtest_neg10_labels_dst = th.LongTensor(np.load('dataset/{}_outtest/{}_neg10_labels_dst.npy'.format(args.outtest,args.outtest)))
    outtest_neg10_labels_src = th.LongTensor(np.load('dataset/{}_outtest/{}_neg10_labels_src.npy'.format(args.outtest,args.outtest)))


    ### load model
    if (args.device >= 0) and (th.cuda.is_available()):
        model = th.load(args.model,map_location='cuda:{}'.format(args.device))
    else:
        model = th.load(args.model,map_location='cpu')

    outtest_accuracy, outtest_auprc, outtest_auroc = evaluate(model, g_all, outtest_tar_nid, node_features_all, edge_weights_all,
                                                          args.batch_size, outtest_pos_labels_src, outtest_pos_labels_dst,
                                                          outtest_neg10_labels_src, outtest_neg10_labels_dst)
    print(args.outtest+'_neg10:', ' accuracy: ', round(outtest_accuracy,3), '| auprc:', round(outtest_auprc,3), 'auroc:', round(outtest_auroc,3))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if (args.device >= 0) and (th.cuda.is_available()):
        device = th.device('cuda:{}'.format(args.device))
    else:
        device = th.device('cpu')
    outtest_predict(args)
    



















