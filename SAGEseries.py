import time
from scipy.fftpack import shift
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GCN2Conv, ChebConv, ARMAConv, APPNP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import networkx as nx
from torch_geometric.utils.undirected import is_undirected, to_undirected


#only for message passing
class passing(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index = None, edge_weight = None, adj_t = None):
        if adj_t is not None:
            return self.propagate(edge_index=adj_t, x=x)
        else:    
            return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
#only for message passing

class GraphSAGE_RW_full(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GraphSAGE_RW_full, self).__init__()
        data = dataset[0]
        #This modification is for cora only.
        #Please use looped version for other datasets.
        #Maybe better to other datasets, but unnecessary.
        
        data.edge_index = add_remaining_self_loops(to_undirected(data.edge_index))[0]
        #data.edge_index = to_undirected(data.edge_index)
        
        self.num_features = dataset.num_features
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.K = args.K
        self.nlayer = args.nlayer
        self.aug = args.sage_aug
        
        self.lins = ModuleList()
        self.lins.append(Linear(self.num_features * 2, self.hidden))
        for i in range(args.nlayer - 1):
            self.lins.append(Linear(self.hidden * 2, self.hidden))
        self.lins.append(Linear(self.hidden, dataset.num_classes))
        
        tmptensor = args.alpha * ((1-args.alpha) ** torch.arange(0, args.K+1))
        tmptensor[-1] = (1-args.alpha) ** args.K
        tmptensor = tmptensor.repeat(self.nlayer, 1)
        self.att = Parameter(tmptensor)

        # rws settings 
        self.rws = args.rws
        self.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

        #graph information
        self.N = data.num_nodes
        self.edge_index = data.edge_index
        self.adj = SparseTensor(
            row = data.edge_index[0],
            col = data.edge_index[1],
            sparse_sizes = (data.num_nodes, data.num_nodes)
        )
        self.degree = sparsesum(self.adj, dim=0)
        self.weight = (self.degree[self.edge_index[0]] ** -0.5) * (self.degree[self.edge_index[1]] ** -0.5)
        
        self.passer = passing()

        self.reset_parameters()
        self.move()
        
    def reset_parameters(self):
        for i in range(self.nlayer + 1):
            self.lins[i].reset_parameters()

    
    def move(self):
        self.passer = self.passer.to(self.device)
        self.degree = self.degree.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        self.weight = self.weight.to(self.device)
        self.adj = self.adj.to(self.device)

    def samp_aggr_layer(self, x, edge_index, layer_index):
        batch = torch.arange(0, self.N, device = self.device)
        mask = self.att[layer_index]
        if self.training:
            batch = torch.arange(0, self.N, device = self.device)
            batchNum = len(batch)
            features = len(x[0])
            aggx = x[batch] * mask[0]
            aug = x[batch]
            for i in range(1, self.K+1):
                startIndex = torch.arange(0, batchNum * self.rws, device = self.device) // self.rws
                starts     = batch[startIndex]
                ends       = self.adj.random_walk(starts, i)
                ends       = ends[:,-1].flatten()
                aggx_      =(((self.degree[starts] ** 0.5) * (self.degree[ends] ** -0.5)).unsqueeze(1) * x[ends])\
                            .reshape(batchNum, self.rws, features).mean(dim=1)
                if self.aug == i:
                    aug = aggx_[batch]
                aggx += aggx_ * mask[i]
                
            aggx = torch.cat((aug, aggx), dim = 1)
            return aggx
        else:
            aug = x[batch]
            aggx = x[batch] * mask[0]
            for i in range(1, self.K+1):
                x = self.passer(x, self.edge_index, self.weight)
                if self.aug == i:
                    aug = x[batch]
                aggx += x[batch] * mask[i]
            aggx = torch.cat((aug, aggx), dim = 1)
            return aggx



    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.nlayer):
            x = self.samp_aggr_layer(x, edge_index, i)
            x = F.relu(self.lins[i](F.dropout(x, p = self.dropout, training = self.training)))
            #x = F.normalize(x, p = 2, dim = 1)
        x = self.lins[-1](F.dropout(x, p = self.dropout, training = self.training))
        return F.log_softmax(x, dim=1)
