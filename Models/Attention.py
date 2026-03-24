import torch
from torch import nn
from utils import Arguments
args = Arguments.Arg()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((64, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 64),  # channel, channel // reduction
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # channel // reduction, channel
            nn.Sigmoid()
        )

    def forward(self, x):
        c, _, _ = x.size()#client_number.128.128
        y = self.avg_pool(x)
        y =  y.view(y.size(0), y.size(1))
        y = self.fc(y)
        return y


# 权重计算
class Attention(nn.Module):
    def __init__(self, n):
        super(Attention, self).__init__()
        self.client_num = n
        self.se = SELayer(n, reduction=4)
        self.attention1 = nn.Linear(n, 64)
        self.attention2 = nn.Linear(64, n)


    def forward(self, data):
        x = self.se(data)
        x = x.view(self.client_num)
        x = self.attention1(x)
        x = self.attention2(x)
        attention_weights = torch.softmax(x, dim=0)
        return attention_weights

    def copy(self):
        m = Attention(self.client_num)
        m.load_state_dict(self.state_dict())
        return m


def AttentionNet(client_num):
    return Attention(client_num)