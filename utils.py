import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc



def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU trainers.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])
    

def load_subtensor(features, weights, input_nodes, input_eid_lst):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_features = features[input_nodes]
    
    batch_weights = []
    for eid in input_eid_lst:
        batch_weight = weights[eid]
        batch_weights.append(batch_weight)
    return batch_features, batch_weights


def score_calculate(embs, pos_graph, neg_graph):
    
    with pos_graph.local_scope():
        pos_graph.ndata['h'] = embs
        pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        pos_score = pos_graph.edata['score']
            
    with neg_graph.local_scope():
        neg_graph.ndata['h'] = embs
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']
        
    return pos_score, neg_score

def index_calculate(pos_score, neg_score):
    
    pos_score = th.sigmoid(pos_score)
    neg_score = th.sigmoid(neg_score)
    
    true_pos = th.sum(pos_score >= 0.5).item()
    all_pos = len(pos_score)
    true_neg = th.sum(neg_score < 0.5).item()
    all_neg = len(neg_score)
    accuracy = (true_pos + true_neg) / (all_pos + all_neg)
    
    labels = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).detach().numpy()
    scores = th.cat([pos_score, neg_score]).detach().numpy()
    precision_feat, recall_feat, thresholds_feat = precision_recall_curve(labels, scores)
    auprc = auc(recall_feat, precision_feat)
    auroc = roc_auc_score(labels, scores)
    
    return accuracy, auprc, auroc


def evaluate(model, g, nodes, node_features, edge_weights, batch_size, heads, tails, neg_heads, neg_tails):
    model.eval()
    with th.no_grad():
        preds = model.inference(g, nodes, node_features, edge_weights, batch_size)
    
    embs = th.zeros(g.number_of_nodes(), preds.size()[-1])
    embs[nodes] = preds
    
    pos_graph = dgl.graph((heads, tails), num_nodes=g.number_of_nodes())
    neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=g.number_of_nodes())
    pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
    
    pos_score, neg_score = score_calculate(embs[pos_graph.ndata[dgl.NID]], pos_graph, neg_graph)
    accuracy, auprc, auroc = index_calculate(pos_score, neg_score)
    
    return accuracy, auprc, auroc