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
    
    
class NeighborCollector(object):
    def __init__(self, g, num_layers, heads_all, tails_all, neg_heads_all, num_negs):
        self.g = g
        self.num_layers = num_layers
        self.heads_all = heads_all
        self.tails_all = tails_all
        self.neg_heads_all = neg_heads_all
        self.num_negs = num_negs

    def get_blocks(self, seed_edges):
        
        n_edges = len(seed_edges)
        heads = self.heads_all[seed_edges]
        tails = self.tails_all[seed_edges]
        neg_heads = self.neg_heads_all[seed_edges].view(-1)
        neg_tails = tails.view(-1, 1).expand(n_edges, self.num_negs).flatten()

        # Maintain the correspondence between heads, tails and negative tails as two graphs.
        # pos_graph contains the correspondence between each head and its positive tail.
        # neg_graph contains the correspondence between each head and its negative tails.
        # Both pos_graph and neg_graph are first constructed with the same node space as
        # the original graph.  Then they are compacted together with dgl.compact_graphs.
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        # Obtain the node IDs being used in either pos_graph or neg_graph.  
        # Since they are compacted together, pos_graph and neg_graph share 
        # the same compacted node space.
        seeds = pos_graph.ndata[dgl.NID]
        blocks = []
        for i in range(self.num_layers):
            # For each seed node, get the neighbors.
            frontier = dgl.in_subgraph(self.g, seeds)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            block.edata[dgl.EID] = frontier.edata[dgl.EID]
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return pos_graph, neg_graph, blocks
    

def get_neighbors(g, batch_nodes, layer_num):
    
    blocks = []
    block_eids = []
    seeds = batch_nodes
    
    for i in range(layer_num):
        g_sub = dgl.in_subgraph(g, seeds)
        block = dgl.to_block(g_sub, seeds)
        block_eid = g_sub.edata[dgl.EID]
        
        blocks.insert(0, block)
        block_eids.insert(0, block_eid)
        
        seeds = block.srcdata[dgl.NID]
        
    return blocks, block_eids


class ModelLayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(ModelLayer, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def forward(self, graph, feat, weight):
       
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
        graph.edata['w'] = weight
        graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class Model(nn.Module):
    def __init__(self,
             in_feats,
             n_hidden,
             n_layers,
             activation,
             layer_dropout,
             emb_dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(ModelLayer(in_feats, n_hidden[0], layer_dropout))
        for i in range(1, n_layers):
            self.layers.append(ModelLayer(n_hidden[i-1], n_hidden[i], layer_dropout))
        self.dropout = nn.Dropout(emb_dropout)
        self.activation = activation

    def forward(self, blocks, x, weights):
        h = x
        for l, (layer, block, weight) in enumerate(zip(self.layers, blocks, weights)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst),weight)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, nodes, x, weights, batch_size):

        y = th.zeros(len(nodes), self.n_hidden[-1])
        layer_num = len(self.layers)

        for start in range(0, len(nodes), batch_size):
            
            end = start + batch_size
            batch_nodes = nodes[start:end]
            blocks, block_eids = get_neighbors(g, batch_nodes, layer_num) 
            
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch_features, batch_weights = load_subtensor(x, weights, input_nodes, block_eids)

            h = self.forward(blocks, batch_features, batch_weights)
            y[start:end] = h.cpu()

        return y


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., logits='sigmoid', reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, block_outputs, pos_graph, neg_graph):
        
        pos_score, neg_score = score_calculate(block_outputs, pos_graph, neg_graph)
        scores = th.cat([pos_score, neg_score])
        labels = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
        if self.logits == 'sigmoid':
            BCE_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
        elif self.logits == 'relu':
            # relu not work for backward
            scores = nn.Hardtanh(0., 1.)(scores)
            BCE_loss = F.binary_cross_entropy(scores, labels, reduction='none')
            
        pt = th.exp(-BCE_loss)
        alpha_t = labels*self.alpha + (th.ones_like(labels)-labels)*(1-self.alpha)
        f_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return th.mean(f_loss), pos_score, neg_score
        else:
            return f_loss, pos_score, neg_score
    
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