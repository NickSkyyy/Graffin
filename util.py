from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.datasets import DBLP
from torch_geometric.datasets import Reddit
from torch_geometric.loader import DataLoader
from torch_geometric.utils import *
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

def AUC_ROC(output, y):
  return roc_auc_score(y.detach(), F.softmax(output, dim=-1).detach(), average='macro', multi_class='ovr')

def F1(pred, y):
  return f1_score(y.detach(), pred.detach(), average='macro')

def get_dblp_seq(graph, type="degree", reverse=False):
  node_list = None
  if type == "degree":
    node_list = degree(graph[("author", "to", "paper")].edge_index[0], graph["author"].x.shape[0], dtype=torch.long)
    node_list = torch.argsort(node_list, descending=not reverse)
  sequence_matrix = torch.stack([graph["author"].x[int(line.item())] for line in node_list])
  graph.seq_reverse = sequence_matrix
  graph.seqid_reverse = node_list

  # base info
  graph.x = graph["author"].x
  graph.y = graph["author"].y
  graph.train_mask = graph["author"].train_mask
  graph.test_mask = graph["author"].test_mask
  graph.val_mask = graph["author"].val_mask
  graph.edge_index = graph[("author", "to", "paper")].edge_index
  return graph

def get_sequence(graph, type="degree", reverse=False):
  """
  #### 获取图序列矩阵
  - type: 排序指标，默认度排序
  - reverse: 是否反向，默认升序，考虑序列中最重要节点，反向考虑低资源
  #### 返回
  - sequence_matrix: 排序后的特征矩阵
  - indice: 排序后的索引
  """
  node_list = sort_node(graph, type, reverse)
  sequence_matrix = torch.stack([graph.x[int(line.item())] for line in node_list])
  return sequence_matrix, node_list
def get_sequences(graph, type="degree", reverse=False):
  # sequence_matrix, indice = get_sequence(graph, type, reverse)
  # graph.seq = sequence_matrix
  # graph.seqid = indice
  sequence_matrix, indice = get_sequence(graph, type, not reverse)
  graph.seq_reverse = sequence_matrix
  # row = torch.zeros(len(indice), len(indice))
  # for i, val in enumerate(indice):
  #   row[val][i] = 1
  row = torch.empty_like(indice)
  for i, val in enumerate(indice):
    row[val] = i
  graph.row = row
  # graph.seqid_reverse = row
  graph.seqid_reverse = indice
  # graph.mpnn = graph.x
  return graph

class ClassMask(T.BaseTransform):
  def __call__(self, data: Data) -> Data:
    cnt = torch.bincount(data.y)
    # _, max_index = torch.max(cnt, dim=0)
    _, min_index = torch.min(cnt, dim=0)
    # data.high_mask = torch.tensor([True if y == max_index else False for y in data.y])
    data.low_mask = torch.tensor([True if y == min_index else False for y in data.y])
    # print(cnt)
    # print(max_index)
    # print(min_index)
    data.bin = torch.bincount(data.y)
    ll = len(torch.bincount(data.y))
    data.label_mask = []
    for label in range(ll):
      temp = torch.tensor([True if y == label else False for y in data.y])
      data.label_mask.append(temp)
    data.label_mask = torch.stack(data.label_mask)
    return data

def load(dataset, root=".\\data"):
  print("load dataset %s" % dataset)
  transforms = T.Compose([
    T.RandomNodeSplit(),
    ClassMask()
  ])
  train = None
  if dataset == "DBLP":
    raw = DBLP(root + "\\DBLP")[0]
    paper = raw["paper"].x[:, :334]
    train = Data(x=torch.cat((raw["author"].x, paper), dim=0), 
                 edge_index=raw[("author", "to", "paper")].edge_index,
                 y=raw["author"].y,
                 train_mask = raw["author"].train_mask,
                 test_mask = raw["author"].test_mask,
                 val_mask = raw["author"].val_mask)
    trans = ClassMask()
    train = trans(train)
  elif dataset == "Amazon_Comp":
    train = Amazon(root, "Computers", transforms)[0]
  elif dataset == "Amazon_Photo":
    train = Amazon(root, "Photo", transforms)[0]
  # elif dataset == "Wiki":
  #   train = AttributedGraphDataset(root, "Wiki", transforms)
  elif dataset == "Cora":
    train = AttributedGraphDataset(root, "Cora", transforms)[0]
  # elif dataset == "Reddit":
  #   train = Reddit(root, transforms)
  return train

def sort_node(graph, type="degree", reverse=False):
  """
  #### 节点排序
  - type: 排序指标，默认度排序
  - reverse: 是否反向，默认升序，考虑序列中最重要节点，反向考虑低资源
  #### 返回
  排序节点ID列表
  """
  if type == "degree":
    degrees = degree(graph.edge_index[0], graph.num_nodes, dtype=torch.long)
    degrees = torch.argsort(degrees, descending=reverse)
    return degrees
  elif type == "eigen":
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(graph.x))])
    g.add_edges_from(graph.edge_index.t().contiguous().numpy())
    eigen = nx.eigenvector_centrality_numpy(g)
    eigen = list(eigen.items())
    eigen.sort(key=lambda x : (x[1], x[0]), reverse=reverse)
    eigen = torch.tensor([x[0] for x in eigen])
    return eigen
  print("Warning: Type %s not define, ID order return" % type)
  return torch.arange(0, graph.num_nodes)