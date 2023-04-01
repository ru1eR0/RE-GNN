import argparse
from signal import valid_signals
from dataset_loader import DataLoader
from utils import random_splits, random_splits_citation,fixed_splits,batch_generator
from GCNseries import *
from GCNIIseries import *
from GATseries import *
from SAGEseries import *
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
import math
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import coalesce
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch import pca_lowrank, matmul

def RunExp(args, dataset, data, Net):

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    host = torch.device('cpu')

    def train(model, optimizer, data, dprate):
        
        if args.net in ['GCN_RW_mini_G', 'GraphSAGE_RW_mini_G', 'GCNII_RW_mini_G']:
            batchStack = batch_generator(args.tsplit, data.train_mask, data.num_nodes, args.batch)
            for i in batchStack:
                model.train()
                optimizer.zero_grad()
                out = model(data, i.to(device))[data.train_mask[i]]
                loss = F.nll_loss(out, data.y[i][data.train_mask[i]])
                loss.backward()
                optimizer.step()
                del out
                torch.cuda.empty_cache()
        elif args.net in []:
            None
        else:
            model.train()
            optimizer.zero_grad()
            out = model(data)[data.train_mask]
            loss = F.nll_loss(out, data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            del out
            torch.cuda.empty_cache()

    def test(model, data):
        with torch.no_grad():
            if args.net in ['GCN_RW_mini_G', 'GraphSAGE_RW_mini_G', 'GCNII_RW_mini_G']:
                model.eval()
                logits, accs, losses, preds = model(data, None), [], [], []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                    loss = F.nll_loss(model(data, None)[mask], data.y[mask])
                    preds.append(pred.detach().cpu())
                    accs.append(acc)
                    losses.append(loss.detach().cpu())
                return accs, preds, losses
            elif args.net in []:
                None
            else:
                model.eval()
                logits, accs, losses, preds = model(data), [], [], []
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                    loss = F.nll_loss(model(data)[mask], data.y[mask])
                    preds.append(pred.detach().cpu())
                    accs.append(acc)
                    losses.append(loss.detach().cpu())
                return accs, preds, losses
    
    tmp_net = Net(dataset, args)
    
    model = tmp_net.to(device)
    data = data.to(device)

    if args.net in ['GCN_RW_full', 'GCN_RW_mini_G', 'GraphSAGE_RW_full', 'GCNII_RW_full', 'GCNII_RW_mini_G']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att, 'weight_decay': args.weight_decay, 'lr': args.lr}
        ])
    elif args.net in ['GAT_RW_full']:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)


    if args.net in ['GCN_RW_mini_G', 'GCNII_RW_mini_G']: # fill historical embedding
        with torch.no_grad():
            model.eval()
            model(data, None)
    
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        
        train(model, optimizer, data, args.dprate)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        print(epoch, 'epochs trained. Current Status:', train_acc, val_acc, tmp_test_acc)

        '''
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
        '''
        if val_acc > best_val_acc :
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            if args.net in ['ChebBase','ChebNetII',"ChebNetII_V", 'GPRGNN']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            elif args.net in ['GCN_RW_full', 'GCN_RW_mini_G', 'GraphSAGE_RW_full', 'GraphSAGE_RW_mini_G', 'GCNII_RW_full', 'GCNII_RW_mini_G']:
                theta = tmp_net.att.clone().detach().cpu().flatten().numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, theta, time_run

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--rws', type=int, default=10, help='random walks per node')
    parser.add_argument('--batch', type=int, default=1, help='num of batches')
    parser.add_argument('--ban0', type=bool, default=False, help='ban own embeddings')
    parser.add_argument('--sdim', type=bool, default=False, help='data dim reduction')
    

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--nlayer', type=int, default=2, help='num of network layers')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Chameleon','Squirrel','Actor','Texas','Cornell','Wisconsin','Photo','Computers','ogbn-arxiv', 'ogbn-products', 'wiki', 'pokec', 'twitch-gamer'],default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=5, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GCNII', 'GAT', 'GraphSAGE', 'APPNP', 'ChebNet','MLP','ARMA','GPRGNN','ChebNetII','ChebBase','ChebNetII_V', 'GCN_RW_full', 'GCN_RW_mini_G', 'GraphSAGE_RW_full', 'GraphSAGE_RW_mini_G', 'GAT_RW_full', 'GCNII_RW_full', 'GCNII_RW_mini_G'], default='ChebNetII')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--paraA', type=float, default=0.1, help='Alpha for GCNII')
    parser.add_argument('--paraB', type=float, default=0.1, help='Beta for GCNII')

    parser.add_argument('--tsplit', type=bool, default=False, help='training set splitted only')
    parser.add_argument('--sage_aug', type=int, default=0, help='augmentation of SAGE')
    #parser.add_argument('--osplit', type=bool, default=False, help='maintain original splits')
    #parser.add_argument('--full', type=bool, default=False, help='full-supervise')
    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    #parser.add_argument('--semi_rnd', type=bool, default=False, help='semi random splits')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #10 fixed seeds for random splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == "GCN_RW_full":
        Net = GCN_RW_full
    elif gnn_name == "GCN_RW_mini_G":
        Net = GCN_RW_mini_G
    elif gnn_name == "GCNII_RW_full":
        Net = GCNII_RW_full
    elif gnn_name == "GCNII_RW_mini_G":
        Net = GCNII_RW_mini_G
    elif gnn_name == "GraphSAGE_RW_full":
        Net = GraphSAGE_RW_full
    elif gnn_name == "GAT_RW_full":
        Net = GAT_RW_full
    
    dataset = DataLoader(args.dataset)
    print(dataset.num_classes)
    saver = torch.load('./data_saved/'+args.dataset+'.pt')
    
    data = dataset[0]
    data.edge_index = saver['edge_index']
    data.y = data.y.flatten()
    args.runs = saver['train_mask'].shape[1]
    
    results = []
    time_results=[]
    
    for RP in range(args.runs):
        print("RP", RP, "Launched...")
        args.seed=SEEDS[RP]
        data.train_mask, data.val_mask, data.test_mask = saver['train_mask'][:,RP], saver['val_mask'][:,RP], saver['test_mask'][:,RP]
        test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net)
        print(theta_0)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        #if args.net in ["ChebBase","ChebNetII","ChebNetII_V"]:
        #    print('Weights:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    #print(uncertainty*100)
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    
