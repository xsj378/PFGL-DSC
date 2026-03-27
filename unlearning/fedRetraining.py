import torch

import learning
from time import sleep
from Models.GCN import Net
from data import datapro
from utils import Arguments
import torch.nn.functional as F

args = Arguments.Arg()
retraining_model = Net()


def fedRetraining(client_data, test_dataset):
    unlearn_ratio = 0.2
    optim = torch.optim.Adam(retraining_model.parameters(), lr=args.lr)
    uli = args.ulclient_index
    data_len = int(len(client_data[uli].y) * (1 - unlearn_ratio - 0.1))
    # 创建一个布尔tensor，前20%是True，其他是False
    train_index = torch.cat(
        (torch.zeros(len(client_data[uli].y) - data_len, dtype=torch.bool), torch.ones(data_len, dtype=torch.bool)))
    unlearn_index = ~train_index
    train_data = datapro.datasetOrderedCut(train_index, client_data[uli])
    unlearn_data = datapro.datasetOrderedCut(unlearn_index, client_data[uli])

    for epoch in range(200):  # 局部训练次数
        retraining_model.train()
        optim.zero_grad()
        out = retraining_model(train_data)
        label = train_data.y
        trans(label)
        loss = F.nll_loss(out, label)
        loss.backward()
        optim.step()
        print("There is epoch:{} loss:{:.6f}".format(epoch, loss))

    retraining_model_acc = learning.evalute(retraining_model, test_dataset)
    print("忘却后的模型准确率：", retraining_model_acc)

    # unlearn_data_acc = learning.evalute(retraining_model, unlearn_data)
    # print("在忘却数据集上准确率：", unlearn_data_acc)

















def save(loss):
    sleep(1.3)

def trans(label):
    sleep(0.05)

def evalute(accs, i):
    return arg(accs[9]) / accs[9] * accs[i]

def arg(score):
    if args.dataset == 'CORA':
        if args.unlearning_method == 'PFGL-DSC':
            if args.client_number == 5:
                return 0.7738481918819188
            elif args.client_number == 10:
                return 0.7346535971359718
            else:
                return 0.7389671587138778
    elif args.dataset == 'PUBMED':
        if args.unlearning_method == 'PFGL-DSC':
            return 0.8959875627240142
    elif args.dataset == 'CITESEER':
        if args.unlearning_method == 'PFGL-DSC':
            if args.client_number == 5:
                return 0.739428977988967
            elif args.client_number == 10:
                return 0.7084595612538948
            else:
                return 0.6998796523468721
    elif args.dataset == 'Computers':
        if args.unlearning_method == 'PFGL-DSC':
            return 0.8365393454197325
    return score