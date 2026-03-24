from torch_geometric.datasets import Amazon
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import Arguments


# 有序划分子图
def datasetOrderedCut(index, dataset):
    changed_dataset = Data(
        x=dataset.x[index],
        edge_index=dataset.edge_index[:, index[dataset.edge_index[0]] & index[dataset.edge_index[1]]],
        y=dataset.y[index]
    )
    index_difference = torch.min(changed_dataset.edge_index)
    changed_dataset.edge_index -= index_difference
    return changed_dataset


def Pollution_main():
    arg = Arguments.Arg()
    dataset_name = arg.dataset
    if dataset_name=='CORA':
        # 加载 CORA 数据集
        dataset = Planetoid(root='./data/Cora', name='Cora', transform=NormalizeFeatures())
    if dataset_name=='PUBMED':
        # 加载PubMed数据集
        dataset = Planetoid(root='./data/PubMed', name='PubMed')
    if dataset_name=='CITESEER':
        # 加载CiteSeer数据集
        dataset = Planetoid(root='./data/CiteSeer', name='CiteSeer')
    if dataset_name=='Computers':
        # 加载Computers领域论文数据集
        dataset = Amazon(root='./data/Amazon', name='Computers')
    dataset = dataset[0]
    train_len = int(len(dataset.y) * 0.8)
    # 创建一个布尔tensor，前80%是True，其他是False
    train_index = torch.cat((torch.ones(train_len, dtype=torch.bool), torch.zeros(len(dataset.y) - train_len, dtype=torch.bool)))
    test_index = ~train_index
    train_dataset = datasetOrderedCut(train_index, dataset)
    test_dataset = datasetOrderedCut(test_index, dataset)
    return train_dataset, test_dataset
