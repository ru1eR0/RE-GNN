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

class GAT_RW_full(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_RW_full, self).__init__()
        data = dataset[0]
        data.edge_index = add_remaining_self_loops(to_undirected(data.edge_index))[0]
        
        self.num_features = dataset.num_features
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.K = args.K
        self.nlayer = args.nlayer
        self.sdim = args.sdim

        tmpModuleList=[]
        self.lins = ModuleList()
        self.attl = ModuleList()
        self.attr = ModuleList()
        self.lins.append(Linear(self.num_features, self.hidden))
        self.attl.append(Linear(self.hidden, 1))
        self.attr.append(Linear(self.hidden, 1))
        
        for i in range(args.nlayer - 1):
            self.lins.append(Linear(self.hidden, self.hidden))
            self.attl.append(Linear(self.hidden, 1))
            self.attr.append(Linear(self.hidden, 1))
        self.lins.append(Linear(self.hidden, dataset.num_classes))
        self.LeakyReLU = LeakyReLU(0.2)
        
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
        #self.degree = sparsesum(self.adj, dim=0)
        #self.weight = (self.degree[self.edge_index[0]] ** -0.5) * (self.degree[self.edge_index[1]] ** -0.5)
        
        self.reset_parameters()
        self.move()
        
        self.GRAPHSTORAGER = []
        
    def reset_parameters(self):
        for i in range(self.nlayer + 1):
            self.lins[i].reset_parameters()

    
    def move(self):
        #self.passer = self.passer.to(self.device)
        #self.degree = self.degree.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        #self.weight = self.weight.to(self.device)
        self.adj = self.adj.to(self.device)

    def samp_aggr_layer(self, x, edge_index, layer_index):
        batch = torch.arange(0, self.N, device = self.device)
        batchNum = len(batch)
        mask = self.att[layer_index]
        if self.training:
            x  = self.lins[layer_index](F.dropout(x, p = self.dropout, training = self.training))
            al = self.attl[layer_index](x)
            ar = self.attr[layer_index](x)
            aggx = mask[0] * x[batch]
            
            startIndex = torch.arange(0, batchNum * self.rws, device = self.device) // self.rws
            starts     = batch[startIndex]
            for i in range(1, self.K+1):    
                ends       = self.adj.random_walk(starts, i)
                ends       = ends[:,-1].flatten()
                e_ij       = self.LeakyReLU(al[starts] + ar[ends])
                e_ij       = F.softmax(e_ij.reshape(batchNum, self.rws), dim=1).flatten()
                aggx_      = (e_ij.unsqueeze(1) * x[ends]).reshape(batchNum, self.rws, self.hidden).mean(dim=1)
                aggx      += aggx_ * mask[i]
            
            self.GRAPHSTORAGER.append(ends)
            
            return aggx
        else:
            x  = self.lins[layer_index](F.dropout(x, p = self.dropout, training = self.training))
            al = self.attl[layer_index](x)
            ar = self.attr[layer_index](x)
            aggx = mask[0] * x[batch]
            
            startIndex = torch.arange(0, batchNum * self.rws, device = self.device) // self.rws
            starts     = batch[startIndex]
            for i in range(1, self.K+1):
                aggx_ = torch.zeros(batchNum, self.hidden, device = self.device)
                for j in range(layer_index * self.K + i - 1, len(self.GRAPHSTORAGER), self.nlayer * self.K):
                    ends    = self.GRAPHSTORAGER[j]
                    e_ij    = self.LeakyReLU(al[starts] + ar[ends])
                    e_ij    = F.softmax(e_ij.reshape(batchNum, self.rws), dim=1).flatten()
                    aggx_  += (e_ij.unsqueeze(1) * x[ends]).reshape(batchNum, self.rws, self.hidden).mean(dim=1)
                aggx_ /= (len(self.GRAPHSTORAGER) / self.nlayer / self.K)
                aggx  += aggx_ * mask[i]
            return aggx
                    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #if self.sdim:
        #    x = self.sdimlin(F.dropout(x, p = self.dropout, training = self.training))
        #x = F.relu(self.lin0(F.dropout(x, p = self.dprate, training = self.training)))

        for i in range(self.nlayer):
            x = self.samp_aggr_layer(x, edge_index, i)
            #x = F.relu(self.lins[i](F.dropout(x, p = self.dprate, training = self.training)))
            #x = F.normalize(x, p = 2, dim = 1)

        x =        self.lins[-1](F.dropout(x, p = self.dropout, training = self.training))
        return F.log_softmax(x, dim=1)
