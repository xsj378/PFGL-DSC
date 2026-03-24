"""
偏好反学习（PFU）——对应论文 method.tex Algorithm 2 与 Preference Unlearning 小节。

要点：
- 偏好识别：用个性化模型在整图 G_n 上算节点 NLL 损失，按 equ:partitionnodes 以 τ 划分 G_n^p / G_n^u。
- 数据忘却：仅在非偏好子图 G_n^u 上优化 L_total = α(L_nll+L_eucl)+(1-α)L_ref（最小化该损失等价于
  对非偏好数据抬高 NLL，同时用 L_eucl 将参数拉向参考模型 W^ref）。
- 恢复训练：在偏好子图 G_n^p 上常规模拟合，修复忘却带来的精度损失（论文「Resume training」）。
"""
import torch
import torch.nn.functional as F
from algorithm.similarity_cal import att_cal
from data import datapro
from learning import evalute, personalized_aggregation
from utils.ModelUtils import *
from Models import GCN
from torch_geometric.data import Data
from .fedRetraining import evalute as evalutee


args = Arguments.Arg()
Net = GCN.Net()
r = 2.5


# 划分子图
def datasetUnorderedCut(index, dataset):
    changed_dataset = Data(
        x=dataset.x[index],
        edge_index=dataset.edge_index[:, index[dataset.edge_index[0]] & index[dataset.edge_index[1]]],
        y=dataset.y[index]
    )
    # 记录边的数量
    edge_num = len(changed_dataset.edge_index[0])
    # 创建节点到新索引的映射
    index_map = torch.arange(len(index))[index].tolist()
    for i in range(edge_num):
        changed_dataset.edge_index[0][i] = index_map.index(changed_dataset.edge_index[0][i])
    for i in range(edge_num):
        changed_dataset.edge_index[1][i] = index_map.index(changed_dataset.edge_index[1][i])
    return changed_dataset


def compute_node_outputs_and_loss_vector(data: Data, model):
    """
    论文式(4.12)(4.13) 的实现骨架：
    第2行 — 用个性化模型 M_n^p 在整图 G_n 上前向，得到各节点输出（此处为各类别 log 概率，与 log P(y|x) 一致）；
    第3行 — 构造逐节点损失向量 L，L_i = -log Y_i(Q_i)，其中 Q_i 为节点 i 的真实类；
            对 log_softmax 输出等价于 F.nll_loss(..., reduction='none')。
    返回 (node_outputs_log_prob, loss_vector)，二者形状分别为 [|V_n|, Q]、[|V_n|]。
    """
    model.eval()
    with torch.no_grad():
        node_outputs_log_prob = model(data)
    loss_vector = F.nll_loss(node_outputs_log_prob, data.y, reduction="none")
    return node_outputs_log_prob, loss_vector


# 根据用户偏好将图数据划分成两张子图：与用户偏好相似和不相似的子图
def split_graph_by_preference(data, model, similarity_threshold=0.5):
    """
    :param data: 原始图数据，形式为 Data(x, edge_index, y)
    :param model: 客户端的图模型 M_n^p
    :param similarity_threshold: 论文式(4.14) 中的阈值 τ（L_i > τ 的节点归入偏好子图 G_n^p）
    :return: (G_n^p, G_n^u) 对应的 boost_data, unlearn_data
    """
    data_len = len(data.y)
    # 算法4.2 第2-4行：全图前向 + 损失向量 L（式4.12、4.13）
    _, loss_vector = compute_node_outputs_and_loss_vector(data, model)
    # 算法4.2 第5-6行：式(4.14) V_n^p = {i ∈ V_n | L_i > τ}，V_n^u = V_n \\ V_n^p，并诱导边集
    preference_mask = torch.zeros(data_len, dtype=torch.bool, device=loss_vector.device)
    preference_mask[loss_vector > similarity_threshold] = True
    boost_data = datasetUnorderedCut(preference_mask, data)
    unlearn_data = datasetUnorderedCut(~preference_mask, data)
    return boost_data, unlearn_data


# 梯度上升
def gradient_up(model, lr):
    name_lin = ["lin1.weight", "lin2.weight", "lin3.weight", "lin1.bias", "lin2.bias", "lin3.bias"]
    for name, param in model.named_parameters():
        if name in name_lin:
            if param.grad is not None:
                model.state_dict()[name].copy_(param.data + lr * 250 * param.grad.data)


#  限制模型参数不远离参考模型
def gradient_up_1(model, Wref, lr, a):
    name_lin = ["lin1.weight", "lin2.weight", "lin3.weight", "lin1.bias", "lin2.bias", "lin3.bias"]
    for name, param in model.named_parameters():
        if name in name_lin:
            if param.grad is not None:
                model.state_dict()[name].copy_(
                    (1 - a) * param.data + lr * 500 * param.grad.data + a * Wref.state_dict()[name].data)
        # else:
        #     if param.grad is not None:
        #         model.state_dict()[name].copy_(
        #             param.data + lr * param.grad.data)


def fedavg_updata_weight(model, client_list, cn, n):
    # 修改全局模型参数
    for name, param in model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            data += client_list[i].state_dict()[name].data * cn[i] / n
        with torch.no_grad():
            model.state_dict()[name].copy_(data)
    return model


#  欧氏距离
def l2_penalty(w):
    return ((w ** 2).sum()) ** 0.5


def _euclidean_anchor_loss(Wref, model, lam: float) -> torch.Tensor:
    """论文 L_eucl = -λ Σ_k ||W^ref_k - θ_{n,k}||_2（对含 weight 的参数层求和）。"""
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        wref = Wref.state_dict()[name].data.to(device)
        diff = wref - param
        total = total + torch.norm(diff.reshape(-1), p=2)
    return -lam * total


def unlearn(model: Net, unlearn_data, Wref, lr, Wref_loss: torch.Tensor):
    """
    Algorithm 2 内层循环：在 G_n^u 上最小化
    L_total = α (L_nll + L_eucl) + (1-α) L_ref，
    其中 L_nll = -1/|G_n^u| Σ log P(y|x) 的实现为最小化 (-NLL)，即抬高非偏好上的 NLL。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr)
    lam = getattr(args, "lambda_eucl", 0.05)
    alpha = args.a
    inner = getattr(args, "unlearn_inner_steps", 3)

    for i in range(inner):
        model.train()
        optimizer.zero_grad()
        output = model(unlearn_data)
        label = unlearn_data.y
        # 节点/子图级平均 NLL；取负后作为 L_nll 项以配合「忘却」
        nll_mean = F.nll_loss(output, label)
        L_nll = -nll_mean
        L_eucl = _euclidean_anchor_loss(Wref, model, lam)
        loss = alpha * (L_nll + L_eucl) + (1.0 - alpha) * Wref_loss.detach()
        print("There is unlearn_inner:{} loss:{:.6f} (nll_mean={:.6f})".format(i + 1, loss.item(), nll_mean.item()))
        loss.backward()
        optimizer.step()

    return model


def train(model: Net, preference_data, testdata):
    """论文：在偏好子图 G_n^p 上做标准监督训练以恢复精度。"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    accs = []
    for i in range(args.e):
        model.train()
        opt.zero_grad()
        pred = model(preference_data)
        label = preference_data.y
        loss = F.nll_loss(pred, label)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append(evalute(model, testdata))
    print("---------- 恢复训练 ----------")
    for i in range(args.e):
        print(
            "Recover epoch {:>2}: train loss={:.6f}, 测试集准确率={}".format(
                i + 1, losses[i], evalutee(accs, i)
            )
        )


def federated_unlearning(global_model, client_model, client_data, test_dataset):
    # 算法4.2 输入全局模型、各客户端个性化模型、本地图数据与测试集等。令参考模型 W^ref 为当前全局模型（复制参数）
    Wref = global_model.copy()
    # 算法4.2 前置准备：在参考数据上计算参考损失 L_ref（对应式(4.17)中 ℒ_ref 项）
    Wref_loss = F.nll_loss(Wref(test_dataset), test_dataset.y)
    # 算法4.2 前置准备：将各客户端本地模型参数载入网络结构 M_n^p
    for i in range(len(client_model)):
        modeltmp = GCN.Net()
        modeltmp.load_state_dict(client_model[i])
        client_model[i] = modeltmp
    print("--------------------client--------------------")
    _, labels = att_cal(client_model)
    print("----------------忘却训练:------------------")
    # 算法4.2 前置准备：确定执行忘却的客户端编号 n 及非偏好数据占比等实现参数
    uli = args.ulclient_index
    if getattr(args, "use_preference_split", True):
        # 第2-3行：式(4.12) 全图前向得各节点输出；式(4.13) 构造损失向量 L（见 compute_node_outputs_and_loss_vector）
        # 第4-5行：式(4.14) 据 τ 划分 V_n^p、V_n^u 并导出子图 G_n^p、G_n^u
        tau = getattr(args, "tau_preference", 0.5)
        boost_data, unlearn_data = split_graph_by_preference(
            client_data[uli], client_model[uli], similarity_threshold=tau)
    else:
        # 基线：按节点顺序比例切分（非式4.14 的基于 L 的划分）
        unlearn_ratio = 0.5
        data_len = int(len(client_data[uli].y) * (unlearn_ratio + 0.1))
        # 创建一个布尔tensor，前20%是True，其他是False
        unlearn_index = torch.cat(
            (torch.ones(data_len, dtype=torch.bool), torch.zeros(len(client_data[uli].y) - data_len, dtype=torch.bool)))
        boost_index = ~unlearn_index
        unlearn_data = datapro.datasetOrderedCut(unlearn_index, client_data[uli])
        boost_data = datapro.datasetOrderedCut(boost_index, client_data[uli])
    # 进行训练，让模型离要忘却的模型距离远
    # 算法4.2 第7-13行：对 t = 1…T_unlearn，在非偏好子图 G_n^u 上按式(4.17)优化 L_total，更新 M_n^p（见 unlearn 内层）
    for epoch in range(1, args.unl_epochs + 1):
        print("---------------epoch{}--------------".format(epoch))
        client_model[uli] = unlearn(client_model[uli], unlearn_data, Wref, args.un_lr, Wref_loss)
    # 评估忘却阶段后本地模型在测试集上的表现
    print("忘却后的模型准确率：", evalute(client_model[uli], test_dataset))
    print("----------------提升训练:------------------")
    # 算法4.2 第14行：在偏好子图 G_n^p 上做恢复训练，修复忘却带来的精度损失
    train(client_model[uli], boost_data, test_dataset)
    with torch.no_grad():
        # 更新权重
        # 算法4.2 第15行：基于更新后的客户端模型重新计算结构相似度矩阵，供簇内加权聚合使用
        adj_matrix, labels = att_cal(client_model)
        # 算法4.2 第15行：个性化聚合更新全局模型与各客户端模型
        global_model, global_change_model, client_model = personalized_aggregation(
            global_model, client_model, global_model.copy(), adj_matrix, labels, test_dataset)
    if args.save_model:
        torch.save(global_model.state_dict(), "unlearning_global_model.pt")
    print("Success")
    return global_model
