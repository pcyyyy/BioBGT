import torch
import community
from torch import nn, optim
from cdlib.utils import convert_graph_formats
from bgt.InfoNCE import InfoNCE
import networkx as nx
import numpy as np
import torch_geometric.nn as gnn

def transition(communities, num_nodes):
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]

def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    com_cs: np.ndarray) -> torch.Tensor:
    edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)

def community_strength(graph, communities):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation,
                 base_model,
                 k: int = 2,
                 skip: bool = False):

        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

def community_InfoNCE(x, adj):
    global loss
    edge_index = torch.nonzero(adj).t()
    graphs = nx.from_numpy_array(adj.numpy())

    # Louvain
    partitions = []
    partition = community.best_partition(graphs)
    partitions.append(partition)

    classified_dict = {}
    for key, value in partitions[0].items():
        if value not in classified_dict:
            classified_dict[value] = [key]
        else:
            classified_dict[value].append(key)

    communities = []
    for key in classified_dict.keys():
        communities.append(classified_dict[key])

    # CED
    com = transition(communities, graphs.number_of_nodes())
    com_cs, node_cs = community_strength(graphs, communities)
    edge_weight = get_edge_weight(edge_index, com, com_cs)
    edge_index_1 = ced(edge_index, edge_weight, p=0.1, threshold=1.0)

    # 参考CSGCL，两层GCN
    encoder1 = Encoder(x.shape[1], 128,
                          torch.nn.PReLU(),
                          base_model=gnn.GCNConv,
                          k=2).to('cpu')
    update_x = x
    augment_x = x
    encoder1.train()
    optimizer = optim.Adam(encoder1.parameters(), lr=0.001)
    criterion = InfoNCE()

    # 训练模型
    for epoch in range(100):
        total_community_loss = 0.0
        #total_community_loss = torch.tensor(total_community_loss, requires_grad=True)
        update_x = encoder1(update_x, edge_index)
        augment_x = encoder1(augment_x, edge_index_1)
        for i in range(len(classified_dict)):
            query = []
            positive = []
            # 生成query、Positive和Negative用于InfoNCE loss
            for j in classified_dict[i]:
                query.append(torch.tensor(update_x[j], dtype=torch.float))
                positive.append(torch.tensor(augment_x[j], dtype=torch.float))
            query = torch.stack(query)
            query.requires_grad = True
            positive = torch.stack(positive)
            positive.requires_grad = True

            # 正样本以外的所有节点，都是负样本
            negative = []
            for k in range(200): # node num
                if k not in classified_dict[i]:
                    negative.append(torch.tensor(update_x[k], dtype=torch.float))
            negative = torch.stack(negative)
            negative.requires_grad = True

            # InfoNCE loss, n个社区的loss相加
            loss = criterion(query, positive, negative)
            total_community_loss += loss

        optimizer.zero_grad()
        total_community_loss.backward()
        optimizer.step()
    return update_x

# calculate Community Contrastive Strategy-based Functional module Extractor
# community_InfoNCE(feature, adjacency_matrix)