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
    parser.add_argument('--model', type=str, default='best_model/final_model.pth')
    
    return parser

def test(args):
    with open('dataset/data_split_dict.json','r') as fin:
        data_split_dict = json.load(fin)
    
    test_nid = th.LongTensor(data_split_dict['test'])
    tar_nid = th.LongTensor(data_split_dict['tar'])
    test_tar_nid = th.cat([test_nid, tar_nid])

    dst_nodes_all = th.LongTensor(np.load('dataset/graph_construct/dst_nodes_all.npy'))
    src_nodes_all = th.LongTensor(np.load('dataset/graph_construct/src_nodes_all.npy'))
    edges_all = (src_nodes_all, dst_nodes_all)

    edge_weights_all = th.Tensor(np.load('dataset/graph_construct/edge_weights_all.npy'))
    node_features_all = th.Tensor(np.load('dataset/graph_construct/node_features_all.npy'))

    edge_weights_all = edge_weights_all.to(device)
    node_features_all = node_features_all.to(device)
    g = dgl.graph(edges_all)

    test_pos_labels_dst = th.LongTensor(np.load('dataset/labels/test_pos_labels_dst.npy'))
    test_pos_labels_src = th.LongTensor(np.load('dataset/labels/test_pos_labels_src.npy'))
    test_neg10_labels_dst = th.LongTensor(np.load('dataset/labels/neg_samples/test_neg10_labels_dst.npy'))
    test_neg10_labels_src = th.LongTensor(np.load('dataset/labels/neg_samples/test_neg10_labels_src.npy'))
    
    if (args.device >= 0) and (th.cuda.is_available()):
        model = th.load(args.model, map_location='cuda:{}'.format(args.device))
    else:
        model = th.load(args.model,map_location='cpu')
    
    test_accuracy, test_auprc, test_auroc = evaluate(model, g, test_tar_nid, node_features_all, edge_weights_all,
                                                      args.batch_size, test_pos_labels_src, test_pos_labels_dst,
                                                      test_neg10_labels_src, test_neg10_labels_dst)
    
    print('Test_neg10_accuracy: ', round(test_accuracy,3),'| Test_auprc:', round(test_auprc,3), '| Test_auroc:', round(test_auroc,3))
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    
    if (args.device >= 0) and (th.cuda.is_available()):
        device = th.device('cuda:{}'.format(args.device))
    else:
        device = th.device('cpu')
    test(args)
    

