import numpy as np
import torch
from Models import GCN
from Models.Attention import AttentionNet
from utils import Arguments
from Models.GCN import Net

args = Arguments.Arg()
client_number = args.client_number
model = Net()

def unlearning_DFUL(global_model,Var,epochs,idx,unlearning_ratio):

    local_model_set = Var.get_var("client_model")
    change_models = Var.get_var("global_change_model")

    acc_ratio = Var.get_var("acc_ratio")
    att_ratio = Var.get_var("att_ratio")
    client_ratio_globle = Var.get_var("client_ratio_globle")
    client_ratio_globle = client_ratio_globle[-1][idx[-1]:]
    for local_epoch in range(epochs-1,epochs):
        local_t = local_model_set[local_epoch]
        # 删除操作
        # delUserModel = GCN.Net()
        # delUserModel.load_state_dict(local_t[idx])
        # model1 = model_neg(delUserModel)
        # global_model = agg_model_plus(model1, Var, global_model)
        client_new_number = client_number-len(idx)
        weight_name = ["pool1.weight", "pool2.weight", "pool3.weight", "lin1.weight", "lin2.weight", "lin3.weight"]
        client_model_pool_weight = []
        for i in range(len(weight_name)):
            client_model_pool_weight.append([])
        client_att_ratio = [0 for _ in range(client_new_number)]
        acc_dataset_ratio = acc_ratio[local_epoch][idx[-1]:]
        # Max_Min
        acc_dataset_ratio_max = max(acc_dataset_ratio)
        acc_dataset_ratio_min = min(acc_dataset_ratio)
        for kcrm in range(client_new_number):
            acc_dataset_ratio[kcrm] = (acc_dataset_ratio[kcrm] - acc_dataset_ratio_min) / (
                        acc_dataset_ratio_max - acc_dataset_ratio_min)
        acc_dataset_ratio_sum = sum(acc_dataset_ratio)
        for kcrm in range(client_new_number):
            acc_dataset_ratio[kcrm] = acc_dataset_ratio[kcrm] / acc_dataset_ratio_sum

        client_list = local_t[idx[-1]:]
        for i in range(len(client_list)):
            modeltmp = GCN.Net()
            modeltmp.load_state_dict(client_list[i])
            client_list[i] = modeltmp
        for i in range(client_new_number):
            for name, param in client_list[i].named_parameters():
                for j in range(len(weight_name)):
                    if name in weight_name[j]:
                        client_model_pool_weight[j].append(param.data)
        for i in range(len(weight_name)):
            AttentionModel = AttentionNet(int(client_number * (1 - unlearning_ratio)))
            param_list = []
            for j in range(client_new_number):
                param_list.append(client_model_pool_weight[i][j])
            Param = torch.stack(param_list, dim=0)
            Param = Param - Param.mean()
            client_ratio = AttentionModel(Param)
            client_ratio = client_ratio.tolist()
            for kcrs in range(client_new_number):
                client_att_ratio[kcrs] = client_att_ratio[kcrs] + client_ratio[kcrs] / len(weight_name)
        for i in range(client_new_number):
            client_ratio_globle[i] = (client_att_ratio[i] * 0.4 + acc_dataset_ratio[i] * 0.6)
        client_ratio_globle_mean = np.mean(client_ratio_globle)
        if client_ratio_globle[acc_dataset_ratio.index(max(acc_dataset_ratio))] < np.mean(client_ratio_globle):
            for j in range(len(client_ratio_globle)):
                # 以均值为对称轴反翻转数据2*0.7,2*0.7-1
                client_ratio_globle[j] = client_ratio_globle_mean * 2 - client_ratio_globle[j]
        client_ratio_globle_reduce = client_ratio_globle - client_ratio_globle_mean
        for i in range(10):
            temp_flag = 0
            for k in range(len(client_ratio_globle)):
                if client_ratio_globle[k] <= (0.5 / int(client_number * (1 - unlearning_ratio))):  # 0.025  IMDB-BINARY 0.1
                    temp_flag = 1
            if temp_flag is 1:
                break
            else:
                client_ratio_globle = client_ratio_globle + client_ratio_globle_reduce  # 扩大差距
        print("修正后的client_ratio", client_ratio_globle)


        global_model, global_change_model, client_model = fedatt_updata_weight(global_model, client_list, model, client_ratio_globle)


        #修正操作
        # tri = 1 / (epochs - i)
        # for j in range(i + 1, epochs):
        #     local_j = local_model_set[j][idx]
        #     global_change_model = change_models[j]
        #     model2 = agg_model_neg(global_change_model, local_model_set[j], tri)
        #     #model2 = agg_model_neg(global_change_model, local_j, epochs, tri)
        #     global_model = model_matrix_cheng(global_model, model2, delUserModel, abs(j-epochs), tri)
        break
    return global_model

def agg_model_plus(model1,Var,model2):
    local_model_set = Var.get_var("client_model")
    change_models = Var.get_var("global_change_model")
    acc_ratio = Var.get_var("acc_ratio")
    att_ratio = Var.get_var("att_ratio")
    client_ratio_globle = Var.get_var("client_ratio_globle")
    for name, param in model1.named_parameters():
        model1.state_dict()[name].copy_((model1.state_dict()[name].data + model2.state_dict()[name].data * client_number) / (client_number - 1))
    return model1

def model_neg(model1):
    model = GCN.Net()
    for name, param in model1.named_parameters():
        model.state_dict()[name].copy_(-model1.state_dict()[name].data)
    return model

# def agg_model_neg(model1,model2,M,tri):
#     model = m_LeNet.Net()
#     for name, param in model1.named_parameters():
#         model.state_dict()[name].copy_((M*model1.state_dict()[name].data-model2.state_dict()[name].data)/(M-1)*tri)
#     return model

def agg_model_neg(model1,local,tri):
    model = GCN.Net()
    a = local[3:]
    for name, param in model1.named_parameters():
        model.state_dict()[name].copy_((len(local)*model1.state_dict()[name].data-a[0].state_dict()[name].data-a[1].state_dict()[name].data-a[2].state_dict()[name].data)/(len(local)-3))
    return model

def model_matrix_cheng(global_model,model1,model2,n,tri):
    model = GCN.Net()
    for name, param in model1.named_parameters():
        #a = model2.state_dict()[name].data/torch.max(torch.abs(model2.state_dict()[name].data)).item()
        norm = torch.abs(model2.state_dict()[name].data)/torch.norm(model2.state_dict()[name].data, p=2)
        #print(torch.norm(model2.state_dict()[name].data, p=2))
        # n = model2.state_dict()[name].data.shape
        # a_flat = model1.state_dict()[name].data.view(n[0], -1)
        # b_flat = model2.state_dict()[name].data.view(n[0], -1)
        # proj = torch.mm(a_flat, a_flat.t())/torch.mm(a_flat.t(), a_flat)
        # print(proj)
        model.state_dict()[name].copy_(global_model.state_dict()[name].data + model1.state_dict()[name].data*norm*n/tri)
    return model
