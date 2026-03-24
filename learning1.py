"""
基于 Attention 权重的联邦图学习（YooChoose 二分类 + BCE）实验脚本。

说明：本文件实现的是按层 Attention 聚合客户端卷积权重的流程，与论文 method.tex 中
PFGL-DSC 的 STE（PCA+曼哈顿 S_p、随机图行为相似度、GAE+KMeans）及 PFU 忘却不在此文件中。
完整 PFGL-DSC 训练/评估请使用 learning.py + algorithm/similarity_cal.py + main.py。
"""
import syft as sy
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import Arguments
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Models.Attention import AttentionNet

# from Models.GCN import Net
#
# model = Net()
hook = sy.TorchHook(torch)
args = Arguments.Arg()
n = 100 / args.client_number
client_ratio_globle = [n for _ in range(args.client_number)]

# def distribute_data_songwei(client_number,tran_right_data,tran_poll_data,model):
#     indx,indy = 0,0
#     right_data, right_targets = tran_right_data
#     total_len = len(right_data)
#     client_data = []
#     client_model = []
#     client_optim = []
#     # n-1个客户端数据均正常
#     for i in range(client_number-1):
#         indx,indy = int(i*total_len/client_number),int((i+1)*total_len/client_number)
#         client_data.append((right_data[indx:indy],right_targets[indx:indy]))
#     client_data.append(tran_poll_data)
#
#     for i in range(client_number):
#         client_model.append(model.copy())
#         client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum))
#     return client_data,client_model,client_optim,model.copy()
def distribute_data_wei(client_number, model):
    client_model = []
    client_optim = []
    for i in range(client_number):
        client_model.append(model.copy())
        client_optim.append(torch.optim.Adam(client_model[i].parameters(), lr=args.lr))
    return client_model, client_optim, model.copy()


def fedatt_updata_weight(model, client_list, global_change_model, client_ratio_sum):
    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data - model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data * client_ratio_sum[i]
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data + model.state_dict()[name].data)  # 加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data)
    # 修改全局模型参数
    for name, param in model.named_parameters():
        data = model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            model.state_dict()[name].copy_(data)

    return model, global_change_model, client_list


def fedavg_updata_weight(model, client_list, global_change_model):
    n = 1 / len(client_list)  # 平均聚合

    # for i in range(len(client_list)):
    #     client_list[i].get()

    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data - model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data
            client_list[i].state_dict()[name].copy_(
                client_list[i].state_dict()[name].data + model.state_dict()[name].data)  # 加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data * n)
    # 修改全局模型参数
    for name, param in model.named_parameters():
        data = model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            model.state_dict()[name].copy_(data)

    return model, global_change_model, client_list


def federated_learning1(client_number, model, client_data, test_data, dataset_ratio, Var):
    print("----------1.1 创建数量为{}的客户端----------".format(client_number))
    client_list = []
    client_data_len = Var.get_var("data_number")
    # for i in range(client_number):
    #     client_list.append(sy.VirtualWorker(hook, id=str(i)))
    print("----------创建成功----------")
    torch.manual_seed(args.seed)
    print("----------1.2 分配模型----------")
    client_model, client_optim, global_change_model = distribute_data_wei(client_number, model)
    # print(client_model[0]==model)
    print("----------分配成功----------")
    print("----------1.3 开始训练----------")
    for i in range(client_number):
        client_model[i].train()
        client_list.append(client_model[i])

    # print(client_list[0]._objects)

    # client_model[i].send(client_list[i])
    # print(client_list[0]._objects)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.BCELoss()

    for i in range(client_number):
        print("----------第{}个客户端开启训练----------".format(i + 1))

        k = 0
        mean_loss = 0
        for epoch in range(args.local_epochs):  # 局部训练次数
            j = 0
            loss_client = 0
            k = k + 1
            value = 0
            client_epochs_loc = 0
            for data in client_data[i]:
                # print(data)    # DataBatch(x=[232, 1], edge_index=[2, 242], y=[64], batch=[232], ptr=[65])
                # DataBatch(x=[194, 1], edge_index=[2, 203], y=[64], batch=[194], ptr=[65])
                client_epochs_loc = client_epochs_loc + 1
                # data = data
                # optimizer.zero_grad()
                client_optim[i].zero_grad()

                # print(model, client_model[i], model == client_model[i])
                pred = client_model[i](data)

                # pred = client_model[i](data)
                label = data.y

                loss = crit(pred, label)
                loss.backward()
                loss_client += data.num_graphs * loss.item()

                # optimizer.step()
                client_optim[i].step()

                # print('训练数据:', client_epochs_loc)
                # print("loss:", loss.data)   # loss: tensor(0.7096)

                # if k % 100 == 0:
            print("There is epoch:{} loss:{:.6f}".format(k, loss_client / client_data_len[i]))
            value = loss_client / client_data_len[i]  # 当前轮次的损失
            mean_loss = mean_loss + value
            if epoch == 0:
                Var.insert_var("user_loss_start_epoch", value, format="list", epoch=True)
            if epoch == args.local_epochs - 1:
                Var.insert_var("user_loss_end_epoch", value, format="list", epoch=True)
        Var.insert_var("train_loss_epoch", mean_loss / epoch, format="list", epoch=True)
    with torch.no_grad():
        # 更新权重
        # model, global_change_model, client_model = fedavg_updata_weight(model, client_model, global_change_model)
        # acc = evalute(model, test_data)
        # print("avg算法下模型准确率：", acc)

        client_model_pool_weight = [[], [], [], [], [], [], [], [], [], [], [], []]
        client_ratio_sum = [0 for _ in range(client_number)]
        weight_name = ["conv1.lin_l.weight", "conv1.lin_r.weight", "pool1.weight", "conv2.lin_l.weight",
                       "conv2.lin_r.weight", "pool2.weight", "conv3.lin_l.weight", "conv3.lin_r.weight", "pool3.weight",
                       "lin1.weight", "lin2.weight", "lin3.weight"]
        for i in range(len(client_list)):
            for name, param in client_list[i].named_parameters():
                for j in range(12):
                    if name in weight_name[j]:
                        client_model_pool_weight[j].append(param.data)

        for i in range(12):
            AttentionModel = AttentionNet()
            param_list = []
            for j in range(client_number):
                param_list.append(client_model_pool_weight[i][j])
                # print(client_model_pool_weight[i][j].size())
            Param = torch.stack(param_list, dim=0)
            # print(Param.size())
            client_ratio = AttentionModel(Param)
            for kcrs in range(client_number):
                client_ratio_sum[kcrs] = client_ratio_sum[kcrs] + client_ratio[kcrs] / 12
        print("Attention后的权重比例：", client_ratio_sum)

        for i in range(client_number):
            client_ratio_sum[i] = client_ratio_sum[i]  # * 0.35+(dataset_ratio[i+1]-dataset_ratio[i]) * 0.65 / 100
            client_ratio_globle[i] = (client_ratio_sum[i] + client_ratio_globle[i]) / 2
        print("client_ratio:    ", client_ratio_sum)

        model, global_change_model, client_model = fedatt_updata_weight(model, client_model, global_change_model,
                                                                           client_ratio_sum)
        acc = evalute(model, test_data)
        print("att算法下模型准确率：", acc)
        client_model_state_dict = []
        for client_model_i in range(client_number):
            client_model_state_dict.append(client_model[client_model_i].state_dict())
        Var.insert_var("client_model", client_model_state_dict, format="list")
        Var.insert_var("acc", acc, "list")
        Var.insert_var("global_change_model", global_change_model.state_dict(), format="list")

    return model
