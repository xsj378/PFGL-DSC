embed_dim = 128

from torch_geometric.nn import GCNConv, GraphConv
from utils import Arguments
import torch.nn.functional as F
import torch

args = Arguments.Arg()


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(args.num_node_features, 64)
        self.conv2 = GCNConv(64, args.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def copy(self):
        m = GCNNet()
        m.load_state_dict(self.state_dict())
        return m


class STRGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(STRGCN, self).__init__()
        self.conv1 = GraphConv(num_features, 32)
        self.conv2 = GraphConv(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

    def copy(self):
        m = STRGCN(self.conv1.in_channels, self.conv2.out_channels)
        m.load_state_dict(self.state_dict())
        return m


def Net():
    return GCNNet()


def STRGCNNet(in_channels, out_channels):
    return STRGCN(in_channels, out_channels)