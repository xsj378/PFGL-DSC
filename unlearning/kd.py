import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from time import sleep
from data import datapro
from unlearning.fedRetraining import save
from utils import Arguments
import torch.nn.functional as F

args = Arguments.Arg()


# def test(model, loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in loader:
#             # data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#         test_loss /= len(loader.dataset)
#         print('Test set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)'.format(
#             test_loss, correct, len(loader.dataset),
#             100. * correct / len(loader.dataset)))
#         acc = correct / len(loader.dataset)
#         return acc


def evalute(model, test_data):
    model.eval()

    with torch.no_grad():
        out = model(test_data)
        pred = out.argmax(dim=1)
        label = test_data.y
        correct = int((pred == label).sum().item())
    return correct/len(label)


def erase(model, target_history, test_dataset, n, tar_num):
    for name, param in model.named_parameters():
        for i in range(tar_num):
            data = 0
            for j in range(args.epochs):
                data += target_history[j][-i - 1].state_dict()[name].data - target_history[j][-i - 2].state_dict()[name].data
                model.state_dict()[name].copy_(model.state_dict()[name].data - data / n)
    evalute(model, test_dataset)
    return model


def distillation(t_model, s_model, train_dataset, test_dataset):
    # 准备好预训练好的教师模型
    t_model.eval()
    # 准备学生模型
    s_model.train()
    # 蒸馏温度
    temp = 1
    # hard_loss权重
    alpha = 0.3
    # soft_loss
    soft_loss = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(s_model.parameters(), lr=0.01)

    epoches = 5
    for epoch in range(epoches):
        # 教师模型预测
        with torch.no_grad():
            teacher_preds = t_model(train_dataset)
        # 学生模型预测
        student_preds = s_model(train_dataset)
        targets = train_dataset.y

        # hard_loss
        student_loss = F.nll_loss(student_preds, targets)
        # 计算蒸馏后的预测结果及soft_loss
        distillation_loss = soft_loss(
            F.log_softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )

        # 将 hard_loss 和 soft_loss 加权求和
        loss = alpha * student_loss + (1 - alpha) * distillation_loss
        save(loss)
        # loss = distillation_loss
        # loss = zhihu_loss

        # 反向传播,优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s_model.eval()

        with torch.no_grad():
            out = s_model(train_dataset)
            label = train_dataset.y
            loss = F.nll_loss(out, label)
            loss += loss.item()
        print("学生模型准确率：", evalute(s_model, test_dataset))
        s_model.train()
    return s_model


def k_d(model, target_history, client_data, test_data, n, target_number):
    unlearn_ratio = 0.5
    model_new = model.copy()
    uli = args.ulclient_index
    data_len = int(len(client_data[uli].y) * (1 - unlearn_ratio - 0.1))
    # 创建一个布尔tensor，前20%是True，其他是False
    train_index = torch.cat(
        (torch.zeros(len(client_data[uli].y) - data_len, dtype=torch.bool), torch.ones(data_len, dtype=torch.bool)))
    train_data = datapro.datasetOrderedCut(train_index, client_data[uli])

    print("----------Erase Historical Parameter Updates----------")
    model_new = erase(model_new, target_history, train_data, n, target_number)
    print("----------Remedy With Knowledge Distillation----------")
    model_new = distillation(model, model_new, train_data, test_data)
    print("-------------------Distillation End-------------------")
    print("模型蒸馏后准确率：", evalute(model_new, test_data))
    print("Success")
    return model_new