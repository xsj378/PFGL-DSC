"""
主入口：数据划分（evaluation.tex 各数据集与客户端数由 Arguments 与 datapro 配置）、
联邦训练（learning.federated_learning）、指标记录（acc / train_loss_epoch 写入 Global_Variable 便于画收敛曲线）、
忘却与基线对比（unlearning/fed_unlearning 与 fedRetraining、kd 等）。
"""
import random
import time

import torch
import numpy as np
from Models.GCN import Net
from data import datapro
from unlearning import kd, fedRetraining
import learning
from unlearning.fed_unlearning import federated_unlearning
from utils.DataUtils import DataUtils, Global_Variable
from utils.ModelUtils import *


args = Arguments.Arg()
client_number = args.client_number
Var = Global_Variable(client_number)
dataUtils = DataUtils()
modelUtils = ModelUtils()
model = Net()
epoch = args.epochs


def save_data(Var):
    # dataUtils.insert_data_to_excel("train_loss_epoch.xlsx", Var.get_var("train_loss_epoch")[-1], sheet_name="Sheet")
    dataUtils.insert_data_to_excel("acc.xlsx", [Var.get_var("acc")[-1]], sheet_name="Sheet")
    # dataUtils.insert_data_to_excel("user_loss_start_epoch.xlsx", Var.get_var("user_loss_start_epoch")[-1],
    #                                sheet_name="Sheet")
    # dataUtils.insert_data_to_excel("user_loss_end_epoch.xlsx", Var.get_var("user_loss_end_epoch")[-1],
    #                                sheet_name="Sheet")


def distribute_data(client_number, train_dataset, dataset_ratio, Var):
    total_len = len(train_dataset.y)
    client_data = []
    client_data_len = []

    # 计算缺失边比例
    # total_edge_num = len(train_dataset.edge_index[0])
    # local_edge_num = 0.0

    for i in range(client_number):
        indx, indy = int(dataset_ratio[i] * total_len / 100), int(
            dataset_ratio[i + 1] * total_len / 100)
        train_index = torch.full((total_len,), False, dtype=torch.bool)
        train_index[indx:indy] = True
        data = datapro.datasetOrderedCut(train_index, train_dataset)
        # local_edge_num += len(data.edge_index[0])
        # print(i+1,":",len(data.edge_index[0]))
        client_data.append(data)
        client_data_len.append(len(data.y))
    Var.insert_var("data_number", client_data_len)
    # print("缺失边比例：", (1 - local_edge_num/total_edge_num)*100)
    return client_data


def MAE(test_model, test_data):
    test_model.eval()
    with torch.no_grad():
        out = test_model(test_data)
        pred = out.argmax(dim=1) + 1 # 使标签号从1开始
        label = test_data.y + 1
        MAE = float(sum(abs(label - pred))) / len(label)
        MAPE = sum(abs(label - pred).float() / label) / len(label) * 100
        RMSE = pow(float(sum((label - pred) ** 2)) / len(label), 1/2)
    return MAE, MAPE, RMSE


if __name__ == '__main__':
    # random.seed(7)# 随机种子
    # dataset_ratio = random.sample(range(1, 100), client_number - 1)
    # dataset_ratio.append(0)
    # dataset_ratio.sort()# 客户端数据量比例分配
    # dataset_ratio.append(100)
    # dataset_ratio = [0, 8, 15, 22, 33, 47, 55, 64, 73, 89, 100]
    dataset_ratio = [int(100*i/client_number) for i in range(client_number+1)] # 平均划分
    # print(dataset_ratio)
    print("----------1.处理数据阶段----------")
    train_dataset, test_dataset = datapro.Pollution_main()
    print("----------2.分发数据阶段----------")
    client_data = distribute_data(client_number, train_dataset, dataset_ratio, Var)

    print("----------2.联邦学习阶段----------")
    n = 1 / args.client_number
    model_list = [model.copy() for _ in range(args.client_number)]
    global_model = model.copy()
    for i in range(epoch):
        print("----------epoch={}----------".format(i + 1))
        model_list, global_model = learning.federated_learning(client_number, model_list, global_model, client_data, test_dataset, Var)
        save_data(Var)

    acc = learning.evalute(global_model, test_dataset)
    print("基于测试集的模型准确率：{}".format(acc))

    print("----------3.模型保存----------")
    modelUtils.save_model("output-0.17.pt", global_model)
    dataUtils.save_var("var-0.17", Var)

    # print("----------4.影响计算----------")
    # model = modelUtils.load_model("output-0.17.pt",global_model)
    # Var = dataUtils.load_var("var-0.17")
    # test_acc = learning.evalute(global_model, test_dataset)
    # print("基于测试集的模型准确率: ",test_acc)
    # impact = ca_impact(Var)
    # print(impact)

    print("----------5.忘却学习阶段----------")
    global_model = modelUtils.load_model("output-0.17.pt", model)
    Var = dataUtils.load_var("var-0.17")
    test_acc = learning.evalute(global_model, test_dataset)
    print("基于测试集的模型准确率: ", test_acc)

    # 开始计时
    start_time = time.time()

    # PFGL-DSC
    if(args.unlearning_method=='PFGL-DSC'):
        global_model = federated_unlearning(global_model, Var.get_var("client_model")[-1][:], client_data, test_dataset)

    # FedRetraining
    if (args.unlearning_method == 'FedRetraining'):
        retraining_model = fedRetraining.fedRetraining(client_data, test_dataset)

    # KD
    if (args.unlearning_method == 'KD'):
        target_history = []
        target_number = 1
        client_model = Var.get_var("client_model")
        for i in range(epoch):
            history_con = []
            for j in range(client_number):
                modeltmp = Net()
                modeltmp.load_state_dict(client_model[i][j])
                history_con.append(modeltmp)
            target_history.append(history_con)
        global_model = kd.k_d(global_model, target_history, client_data, test_dataset, sum(Var.get_var("data_number")), target_number)

    # 结束计时
    end_time = time.time()
    # 计算操作耗时
    elapsed_time = end_time - start_time
    # print("算法耗时：", elapsed_time)


    # # 算法2 识别低质量的数据，梯度上升训练法
    # good_dataset = train_dataset[int(len(train_dataset) * 0.9):]
    # global_model = federated_unlearning(global_model, Var.get_var("client_model")[-1][:-1], [Var.get_var("client_model")[-1][-1]],
    #                        train_poll_dataset, train_dataset, test_dataset, Var.get_var("data_number"), sum(Var.get_var("data_number")))

    # # 算法3 知识蒸馏
    # # target_history = []
    # # target_number = 1
    # # client_model = Var.get_var("client_model")
    # # for i in range(epoch):
    # #     history_con = []
    # #     for j in range(client_number):
    # #         modeltmp = Net()
    # #         modeltmp.load_state_dict(client_model[i][j])
    # #         history_con.append(modeltmp)
    # #     target_history.append(history_con)
    # # global_model = kd.k_d(global_model, target_history, test_dataset, sum(Var.get_var("data_number")), target_number)
