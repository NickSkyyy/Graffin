from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn import GINConv, GINEConv, GPSConv

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreModel(nn.Module):
  def __init__(self, feature_size, hidden_size, num_classes, mpnn_type, dropout, device):
    super(CoreModel, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.device = device

    self.core1 = None
    self.core2 = None

    if mpnn_type == "gnn":
      self.core1 = GraphConv(feature_size, hidden_size)
      self.core2 = GraphConv(hidden_size, num_classes)
    elif mpnn_type == "gcn":
      self.core1 = GCNConv(feature_size, hidden_size)
      self.core2 = GCNConv(hidden_size, num_classes)
    elif mpnn_type == "gat":
      self.core1 = GATConv(feature_size, hidden_size)
      self.core2 = GATConv(hidden_size, num_classes)
    elif mpnn_type == "sage":
      self.core1 = SAGEConv(feature_size, hidden_size)
      self.core2 = SAGEConv(hidden_size, num_classes)

    self.norm = nn.LayerNorm(self.feature_size, device=self.device)
    self.seq = Graffin(self.feature_size, self.hidden_size, dropout, device=self.device)
    self.lin = nn.Linear(self.hidden_size, self.num_classes, device=self.device)

  def forward(self, graph):
    x, edge_index = graph.x, graph.edge_index
    seq_reverse, seqid_reverse = graph.seq_reverse, graph.seqid_reverse
    seq_reverse = self.seq(self.norm(seq_reverse))
    # tseqid_reverse = torch.empty_like(seqid_reverse, device=self.device) 
    # for i, val in enumerate(seqid_reverse):
    #   # tseqid_reverse[val + graph.ptr[graph.batch[i]].item()] = i
    #   tseqid_reverse[val] = i
    # seq_reverse = seq_reverse[tseqid_reverse]
    # seq_reverse = graph.row.matmul(seq_reverse)
    
    x = self.core1(x, edge_index)
    x = F.relu(x)
    x = x * seq_reverse[graph.row]
    x = F.dropout(x, training=self.training)
    # x = self.core2(x, edge_index)
    x = self.lin(x)
    
    return F.log_softmax(x, dim=1)
  
class Graffin(nn.Module):
  def __init__(self, feature_size, hidden_size, dropout, device):
    super(Graffin, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.device = device

    self.llin = nn.Linear(self.feature_size, self.hidden_size, device=self.device)
    self.lfun = nn.GELU()
    self.ldrop = nn.Dropout(dropout)

    self.rlin = nn.Linear(self.feature_size, self.hidden_size, device=self.device)
    self.rgru = nn.GRU(self.hidden_size, self.hidden_size, dropout=dropout, device=self.device)

    self.drop1 = nn.Dropout(dropout)
    # self.flin = nn.Linear(self.hidden_size, self.feature_size, device=self.device)
    # self.drop2 = nn.Dropout(dropout)

  def forward(self, x):
    l = self.ldrop(self.lfun(self.llin(x)))
    r, _ = self.rgru(self.rlin(x))
    # x = self.flin(self.drop1(l * r))
    # return self.drop2(x)
    return self.drop1(l * r)
  
class TestModel(nn.Module):
  def __init__(self, feature_size, hidden_size, num_classes, type, dropout, device):
    super(TestModel, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.device = device

    self.core1 = None
    self.core2 = None

    if type == "gnn":
      self.core1 = GraphConv(feature_size, hidden_size)
      self.core2 = GraphConv(hidden_size, num_classes)
    elif type == "gcn":
      self.core1 = GCNConv(feature_size, hidden_size)
      self.core2 = GCNConv(hidden_size, num_classes)
    elif type == "gat":
      self.core1 = GATConv(feature_size, hidden_size)
      self.core2 = GATConv(hidden_size, num_classes)
    elif type == "sage":
      self.core1 = SAGEConv(feature_size, hidden_size)
      self.core2 = SAGEConv(hidden_size, num_classes)

    self.lin = nn.Linear(hidden_size, num_classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index

    x = self.core1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    # x = self.core2(x, edge_index)
    x = self.lin(x)

    return F.log_softmax(x, dim=1)

# ImbGNN WWW'24
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.nn import Linear
import torch.nn.functional as F
class ImbGNN(torch.nn.Module):
  def __init__(self, feature_size, hidden_size, num_class, use_drop=False):
    super(ImbGNN, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_class = num_class

    self.conv1 = GINConv(
      Sequential(Linear(self.feature_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv2 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv3 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv4 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv5 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))
    
    self.dropout = Dropout(p=0.2)
    self.use_drop = use_drop

    self.lin1 = Linear(self.hidden_size, self.hidden_size)
    self.lin2 = Linear(self.hidden_size, self.num_class)

  def forward(self, graph):
    x, edge_index = graph.x, graph.edge_index
    x = self.conv1(x, edge_index)
    if self.use_drop:
      x = self.dropout(x)
    x = self.conv2(x, edge_index)
    x = self.conv3(x, edge_index)
    x = self.conv4(x, edge_index)
    x = self.conv5(x, edge_index)

    x = self.lin1(x).relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin2(x)

    return F.log_softmax(x, dim=1)
  

class ImbGNNplus(torch.nn.Module):
  def __init__(self, feature_size, hidden_size, num_class, use_drop=False):
    super(ImbGNNplus, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_class = num_class

    self.conv1 = GINConv(
      Sequential(Linear(self.feature_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv2 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv3 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv4 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))

    self.conv5 = GINConv(
      Sequential(Linear(self.hidden_size, self.hidden_size), BatchNorm1d(self.hidden_size), ReLU(),
                  Linear(self.hidden_size, self.hidden_size), ReLU()))
    
    self.dropout = Dropout(p=0.2)
    self.use_drop = use_drop

    self.norm = nn.LayerNorm(self.feature_size, device="cuda")
    self.seq = Graffin(self.feature_size, self.hidden_size, 0.5, device="cuda")

    self.lin1 = Linear(self.hidden_size, self.hidden_size)
    self.lin2 = Linear(self.hidden_size, self.num_class)

  def forward(self, graph):
    x, edge_index = graph.x, graph.edge_index
    seq_reverse, seqid_reverse = graph.seq_reverse, graph.seqid_reverse
    seq_reverse = self.seq(self.norm(seq_reverse))
    tseqid_reverse = torch.empty_like(seqid_reverse, device="cuda") 
    for i, val in enumerate(seqid_reverse):
      # tseqid_reverse[val + graph.ptr[graph.batch[i]].item()] = i
      tseqid_reverse[val] = i
    seq_reverse = seq_reverse[tseqid_reverse]

    x = self.conv1(x, edge_index)
    if self.use_drop:
      x = self.dropout(x)
    x = self.conv2(x, edge_index)
    x = self.conv3(x, edge_index)
    x = self.conv4(x, edge_index)
    x = self.conv5(x, edge_index)

    x = x * seq_reverse

    x = self.lin1(x).relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin2(x)

    return F.log_softmax(x, dim=1)