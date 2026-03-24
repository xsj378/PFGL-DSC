"""
联邦图学习训练与个性化聚合 —— 对应论文 method.tex Algorithm 1 后半段与 Personalized Model Generation。

- aggregation_method='fedatt'：先调用 algorithm.similarity_cal.att_cal 得到结构相似度矩阵 S_s 与聚类 C，
  再 personalized_aggregation 按簇对 S_s 子矩阵 softmax 后加权融合各客户端参数（式 updatepermodel）。
- evaluation.tex 中的准确率、收敛曲线可由 main 循环写入 Global_Variable（acc、train_loss_epoch）再导出。
"""
import copy
import math
import random
import time

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from algorithm.similarity_cal import att_cal
from utils import Arguments
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Models.Attention import AttentionNet
from random import choice
from torch_geometric.data import Data

args = Arguments.Arg()

def evalute(model, test_data):
    model.eval()

    with torch.no_grad():
        out = model(test_data)
        pred = out.argmax(dim=1)
        label = test_data.y
    return float(accuracy_score(label.cpu().numpy(), pred.cpu().numpy()))


def distribute_data(client_number, model_list):
    client_model = []
    client_optim = []
    for i in range(client_number):
        client_model.append(model_list[i].copy())
        client_optim.append(torch.optim.Adam(client_model[i].parameters(), lr=args.lr))
    return client_model, client_optim


def aggregation_weight(global_model, client_list, global_change_model):
    n = 1 / len(client_list)  # 平均聚合
    agg = args.aggregation_method
    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data - global_model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data * n
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data + global_model.state_dict()[name].data)  # 加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data)
    # 修改全局模型参数
    for name, param in global_model.named_parameters():
        data = global_model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            global_model.state_dict()[name].copy_(data)
        if agg=='fedper':
            for i in range(len(client_list)):
                client_list[i].state_dict()[name].copy_(
                    (client_list[i].state_dict()[name].data + global_model.state_dict()[name].data)/2)
        else:
            for i in range(len(client_list)):
                client_list[i].state_dict()[name].copy_(global_model.state_dict()[name].data)
    return global_model, global_change_model, client_list


def personalized_aggregation(global_model, client_list, global_change_model, adj_matrix, labels, test_data):
    """
    论文：对每个簇 c，取 S_s[C_c,C_c]，对称化后按行 softmax 得 S_s^c，再按式
    M_n^p = sum_j Θ_j · S_s^c(n,j) 更新簇内客户端参数；最后对全局模型做均匀平均。
    adj_matrix 即 S_s（可为有向权重，故与转置平均以得到无向聚合权重）。
    """
    n = 1 / len(client_list)
    adj_sum = (adj_matrix + adj_matrix.T) / 2
    # 找出属于同一个聚类的索引
    unique_labels = np.unique(labels)
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    # 将属于同一聚类的元素提取为新的子矩阵
    new_matrices = {}
    for label, indices in clusters.items():
        submatrix = adj_sum[np.ix_(indices, indices)]
        new_matrices[label] = submatrix
    # Alg L12-L13: 对每个簇 c，取 S_s[C_c,C_c]，对称化后按行 softmax 得 S_s^c
    for label, submatrix in new_matrices.items():
        # 对每一行做 softmax 归一化
        adj_normalized = F.softmax(submatrix, dim=1)
        # Alg L14-L16: 再按式 M_n ^ p = sum_j Θ_j · S_s ^ c(n, j) 更新簇内客户端参数；
        client_list = personalized_client_model(client_list, adj_normalized, labels, label)
    # 修改全局模型参数
    for name, param in global_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data - global_model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data * n
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data + global_model.state_dict()[name].data)  # 加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data)
    for name, param in global_model.named_parameters():
        data = global_model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            global_model.state_dict()[name].copy_(data)
    return global_model, global_change_model, client_list


def personalized_client_model(client_list, adj_normalized, labels, label):
    personalized_client_list = copy.deepcopy(client_list)
    # 计算模型的聚合更新
    for name, param in client_list[0].named_parameters():
        adj_x_index = 0
        for i in range(len(client_list)):
            if label!=labels[i]:
                continue
            data = 0
            adj_y_index = 0
            for j in range(len(client_list)):
                if label != labels[j]:
                    continue
                data += client_list[j].state_dict()[name].data * adj_normalized[adj_x_index][adj_y_index]
                adj_y_index += 1
            personalized_client_list[i].state_dict()[name].copy_(data)
            adj_x_index += 1
    return personalized_client_list


def federated_learning(client_number, model_list, global_model, client_data, test_data, Var):
    print("----------1.1 创建数量为{}的客户端----------".format(client_number))
    # 不使用 PySyft TorchHook：未 .send() 到 worker，hook 仅会破坏 torch_geometric 内部 torch.cat 等算子。
    print("----------创建成功----------")
    torch.manual_seed(args.seed)
    print("----------1.2 分配模型----------")
    client_model, client_optim = distribute_data(client_number, model_list)
    global_change_model = global_model.copy()
    print("----------分配成功----------")
    print("----------1.3 开始训练----------")
    for i in range(client_number):
        client_model[i].train()
    client_acc_all = 0
    for i in range(client_number):
        print("----------第{}个客户端开启训练----------".format(i + 1))
        mean_loss = 0
        for epoch in range(args.local_epochs):  # 局部训练次数
            client_model[i].train()
            client_optim[i].zero_grad()
            out = client_model[i](client_data[i])
            label = client_data[i].y
            loss = F.nll_loss(out, label)
            mean_loss += loss.item()
            if args.aggregation_method=='fedprox':
                proximal_term = 0.0
                for w, w_t in zip(client_model[i].parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = F.nll_loss(out, label) + (args.lr / 2) * proximal_term
            loss.backward()
            client_optim[i].step()
            print("There is epoch:{} loss:{:.6f}".format(epoch, loss))
            if epoch == 0:
                Var.insert_var("user_loss_start_epoch", loss.item(), format="list", epoch=True)
            if epoch == args.local_epochs - 1:
                Var.insert_var("user_loss_end_epoch", loss.item(), format="list", epoch=True)
        Var.insert_var("train_loss_epoch", mean_loss / epoch, format="list", epoch=True)
        # client_acc_temp = evalute(client_model[i], test_data)
        # client_acc_all += client_acc_temp
        # print(client_acc_temp)
    # print("本地模型平均准确率:", client_acc_all/client_number)
    with torch.no_grad():
        agg = args.aggregation_method

        # min_acc = []
        # max_acc = []
        # for i in range(client_number):
        #     min_acc.append(evalute(client_model[i], test_data))
        # print("本地模型准确率：", min(min_acc))

        if agg =='PFGL-DSC':
            adj_matrix, labels = att_cal(client_model)
            global_model, global_change_model, client_model = personalized_aggregation(global_model, client_model, global_change_model, adj_matrix, labels, test_data)
        if agg == 'fedavg' or agg == 'fedprox' or agg == 'fedper':
            global_model, global_change_model, client_model = aggregation_weight(global_model, client_model, global_change_model)
        acc = evalute(global_model, test_data)
        print("{}算法下模型准确率：".format(args.aggregation_method), acc)
        client_model_state_dict = []
        for client_model_i in range(client_number):
            client_model_state_dict.append(client_model[client_model_i].state_dict())
        Var.insert_var("client_model", client_model_state_dict, format="list")
        # for i in range(client_number):
        #     max_acc.append(evalute(client_model[i], test_data))
        # print("个性化模型准确率：", max(max_acc))
        # per_acc_avg = 0
        # for i in range(client_number):
        #     per_acc_avg += evalute(client_model[i], test_data)
        # print(per_acc_avg/client_number)
        # acc = [client_acc_all/client_number, acc, per_acc_avg/client_number]
        Var.insert_var("acc", acc, "list")
        Var.insert_var("global_change_model", global_change_model.state_dict(), format="list")
    return client_model, global_model