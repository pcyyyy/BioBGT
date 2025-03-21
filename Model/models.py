# -*- coding: utf-8 -*-
from itertools import product

import torch
from torch import nn
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer
from einops import repeat

class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index, edge_attr=None, degree=None,
            ptr=None, return_attn=False, update_x=None):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                ptr=ptr,
                return_attn=return_attn,
                update_x=update_x
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe_dim=0, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=False, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        self.abs_pe_dim = abs_pe_dim
        if abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        self.embedding_ve = nn.Linear(1, d_model)
        self.embedding_de = nn.Linear(1, d_model)
        self.degree_encoder = nn.Embedding(90, d_model, padding_idx=0)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        kwargs['edge_dim'] = None
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_attn=False):
        x, edge_index, VE = data[0], data[1], data[2]
        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        s = torch.arange(x.shape[0])
        complete_edge_index = torch.vstack((s.repeat_interleave(x.shape[0]), s.repeat(x.shape[0])))
        degree = data.degree if hasattr(data, 'degree') else None

        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))

        #DE = self.embedding_de(degree_matrix)
        #PE = self.embedding_abs_pe(abs_pe)
        VE = self.embedding_ve(VE)
        output = output + VE

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output, 
            edge_index, 
            complete_edge_index,
            edge_attr=None,
            degree=degree,
            ptr=None,
            return_attn=return_attn,
            update_x=data[3]
        )
        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        return self.classifier(output)
