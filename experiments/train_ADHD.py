# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch_geometric.utils as utils
from bgt.models import GraphTransformer
from bgt.utils import count_parameters
from bgt.utils import add_zeros, extract_node_feature
from timeit import default_timer as timer
from dataloader import init_stratified_dataloader
from sklearn.metrics import roc_auc_score, confusion_matrix

def load_args():
    parser = argparse.ArgumentParser(
        description='Biologically Plausible Brain Graph Transformer on ADHD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default="ADHD", help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=128, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--abs-pe-dim', type=int, default=160, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=10, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--edge-dim', type=int, default=128, help='edge features hidden dim')
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}'.format(
            args.dropout, args.lr, args.weight_decay,
            args.num_layers, args.num_heads, args.dim_hidden, bn,
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args

def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()
    running_loss = 0.0
    tic = timer()

    # load node perturbation and extracted feature
    VE = np.load('ADHD_train_NE.npy') #Preprocessing data via node_perturbation.py and FunctionalModuleExtractor.py before loading
    updatex = np.load('ADHD_updatex_only_NE_100epoch.npy')

    for i, data in enumerate(loader):
        size = len(data[2])
        if epoch < args.warmup:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        #if use_cuda:
        #    data = data.cuda()

        output = []
        optimizer.zero_grad()
        for j in range(data[0].shape[0]):
            adj = torch.tensor(data[1][j]).clone()
            adj[adj < 0] = 0
            edge_index = torch.nonzero(adj).t()
            new_data = []
            new_data.append(data[0][j].T)
            new_data.append(edge_index)
            new_data.append(torch.tensor(VE[i*loader.batch_size + j], dtype=torch.float))
            new_data.append(torch.tensor(updatex[i*loader.batch_size + j], dtype=torch.float))
            mid_output = model(new_data)
            dim = -1 if mid_output.dim() == 1 else -2
            mid_output = mid_output.mean(dim=dim, keepdim=mid_output.dim() <= 2)
            output.append(mid_output.squeeze(1))

        output = torch.stack(output)
        loss = criterion(output.squeeze(1), data[2].long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size
    
    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(epoch_loss, toc - tic))
    return epoch_loss

def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    y_pred = []
    y_true = []

    tic = timer()
    # load node perturbation and extracted feature
    VE_eval = np.load('ADHD_val_NE.npy') ##Preprocessing data via node_perturbation.py and FunctionalModuleExtractor.py before loading
    VE_test = np.load('ADHD_test_NE.npy')
    updatex_val = np.load('ADHD_updatex_only_NE_100epoch_val.npy')
    updatex_test = np.load('ADHD_updatex_only_NE_100epoch_test.npy')
    with torch.no_grad():
        count = 0
        for data in loader:
            size = len(data[2])
            #if use_cuda:
            #    data = data.cuda()

            output = []
            for i in range(data[0].shape[0]):
                adj = torch.tensor(data[1][i]).clone()
                adj[adj < 0] = 0
                edge_index = torch.nonzero(adj).t()
                new_data = []
                new_data.append(data[0][i].T)
                new_data.append(edge_index)
                if split == 'Val':
                    new_data.append(torch.tensor(VE_eval[count * size + i], dtype=torch.float))
                    new_data.append(torch.tensor(updatex_val[count * size + i], dtype=torch.float))
                else:
                    new_data.append(torch.tensor(VE_test[count * size + i], dtype=torch.float))
                    new_data.append(torch.tensor(updatex_test[count * size + i], dtype=torch.float))
                mid_output = model(new_data)
                dim = -1 if mid_output.dim() == 1 else -2
                mid_output = mid_output.mean(dim=dim, keepdim=mid_output.dim() <= 2)
                output.append(mid_output.squeeze(1))

            output = torch.stack(output)
            loss = criterion(output.squeeze(1), data[2].long())

            y_true.append(data[2].cpu())
            Temp = output.argmax(dim=-1).view(-1, 1).squeeze(1)
            y_pred.append(Temp.cpu())

            running_loss += loss.item() * size
            count += 1
    toc = timer()
    y_pred = torch.cat(y_pred).unsqueeze(1).numpy()
    y_true = torch.cat(y_true).unsqueeze(1).numpy()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    #evaluator = Evaluator(name=args.dataset)
    #score = evaluator.eval({'y_pred': y_pred,
    #                        'y_true': y_true})['acc']
    score = torch.sum(torch.eq(torch.tensor(y_pred).squeeze(1), torch.tensor(y_true).squeeze(1)).long()).item() / loader.dataset.tensors[0].shape[0]
    print('{} loss: {:.4f} score: {:.4f} time: {:.2f}s'.format(
        split, epoch_loss, score, toc - tic))

    if split == 'Test':
        auroc = roc_auc_score(y_true.squeeze(1), y_pred.squeeze(1))
        print('AUROC: {:.4f}'.format(auroc))

        conf_matrix = confusion_matrix(y_true.squeeze(1), y_pred.squeeze(1))
        tn, fp, fn, tp = conf_matrix.ravel()
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tp = conf_matrix[1, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2 * tp / (2 * tp + fp + fn)

        print('Sensitivity: {:.4f}'.format(sensitivity))
        print('Specificity: {:.4f}'.format(specificity))
        print('F1: {:.4f}'.format(f1_score))

    return score, epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets/'
    data = np.load(data_path+args.dataset+'.npy', allow_pickle=True).item()
    num_edge_features = data["timeseires"].shape[1]
    input_size = num_edge_features
    total_dataloader = init_stratified_dataloader(torch.tensor(data["timeseires"], dtype=torch.float),
                                                torch.tensor(data["adj"], dtype=torch.float),
                                                torch.tensor(data["label"], dtype=torch.float),
                                                data['label'], args.dim_hidden)
    train_loader, val_loader, test_loader = total_dataloader

    model = GraphTransformer(in_size=input_size,
                             num_class=2,
                             d_model=args.dim_hidden,
                             num_heads=args.num_heads,
                             dim_feedforward=2 * args.dim_hidden,
                             dropout=args.dropout,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe_dim=args.abs_pe_dim,
                             num_edge_features=num_edge_features,
                             edge_dim=args.edge_dim,
                             in_embed=False,
                             edge_embed=False,
                             global_pool=args.global_pool)
    #if args.use_cuda:
    #    model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

    lr_steps = args.lr / (args.warmup * len(train_loader))

    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    print("Training...")
    best_val_loss = float('inf')
    best_val_score = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda)
        val_score, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

        if epoch >= args.warmup:
            lr_scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_score'].append(val_score)
        logs['test_score'].append(test_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    total_time = timer() - start_time
    print("best epoch: {} best val score: {:.4f}".format(best_epoch, best_val_score))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

    print("test ACC {:.4f}".format(test_score))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_score': test_score,
            'test_loss': test_loss,
            'val_score': best_val_score,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
             'state_dict': best_weights},
            args.outdir + '/model.pth')

if __name__ == "__main__":
    main()
