"""
结构提取（STE）与客户端结构图构建 —— 对应论文 method.tex Algorithm 1 及式 equ:Sraw、equ:Sp、S_s、GAE+KMeans。

流程摘要（与伪代码一致）：
1. 参数相似度 S_p：各客户端拼接 GCN 卷积层权重 → 按客户端归一化 → PCA 降维 →
   曼哈顿距离得 S_raw(i,j) → 以 S_raw 非对角均值阈值二值化得 S_p。
2. 行为相似度 S_b：在随机图 G_r 上前向，B_n = AVG(M_n(G_r))，再按论文式做 softmax(exp(ξ·B(i,j)))。
3. 结构相似度 S_s = S_p ⊙ S_b（逐元素乘，S_p 作掩码）。
4. 以 S_s 为邻接、客户端参数向量为节点特征，GAE 编码后 KMeans(..., floor(sqrt(N))) 得聚类标签。
"""
import math
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import cosine
from torch_geometric.nn import GAE
from torch_geometric.data import Data
import networkx as nx

from Models.Attention import AttentionNet
from Models.GCN import STRGCNNet
from utils import Arguments

args = Arguments.Arg()


def visualize_clusters(z, labels, method='pca'):
    """evaluation.tex：结构相似度与聚类可视化（潜在空间 2D）。"""
    z_np = z.detach().numpy()

    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method should be either 'pca' or 'tsne'")

    # 将嵌入降到2维
    z_2d = reducer.fit_transform(z_np)
    print(z_2d)
    print(labels)
    # 获取不同簇的标签
    unique_labels = np.unique(labels)

    # 使用tab20色图，这个色图有20种不同的颜色，适用于较多簇的场景
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        plt.scatter(z_2d[labels == label, 0], z_2d[labels == label, 1],
                    color=colors(i), label=f'Cluster {label}', s=100, alpha=0.7, edgecolor='k')
    plt.legend(title="Clusters", loc='best')
    plt.show()

# --- Algorithm 1: 输入：本地模型 M_n、客户端数 N（隐式由 client_model 长度给出）---
def att_cal(client_model):
    start_time = time.time()

    # Algorithm 1 输入：客户端数 N
    client_number = len(client_model)

    # 消融 STE：不使用 S_p/S_b/GAE/KMeans，单簇 + 全 1 邻接；簇内 softmax 后为均匀权重，近似全体客户端同等聚合
    if args.ablation == 'STE':
        labels = np.zeros(client_number, dtype=np.int64)
        adj_matrix = torch.ones(client_number, client_number, dtype=torch.float32)
        return adj_matrix, labels

    weight_name = ["conv1.lin.weight", "conv2.lin.weight"]
    client_model_conv_weight = []
    att_ratio = [0 for _ in range(client_number)]

    # Alg L1-L2: 对每个客户端 n — 提取权重参数 θ_nl 并构建向量 Θ_n（按 conv 层名收集各客户端权重）
    for i in range(len(weight_name)):
        client_model_conv_weight.append([])
    for i in range(client_number):
            for name, param in client_model[i].named_parameters():
                for j in range(len(weight_name)):
                    if name in weight_name[j]:
                        client_model_conv_weight[j].append(param.data)

    # Alg L3: 对每个客户端 Θ_n 归一化 + PCA
    theta_matrix = []
    for n in range(client_number):
        theta_n = []
        for layer_idx in range(len(weight_name)):
            theta_n.append(client_model_conv_weight[layer_idx][n].detach().cpu().view(-1).numpy())
        theta_n = np.concatenate(theta_n, axis=0)
        mu_n, sigma_n = theta_n.mean(), theta_n.std()
        theta_n = (theta_n - mu_n) / (sigma_n + 1e-12)
        theta_matrix.append(theta_n)
    theta_matrix = np.stack(theta_matrix, axis=0)
    n_components = min(client_number, theta_matrix.shape[1])
    theta_pca = PCA(n_components=n_components).fit_transform(theta_matrix)

     # Alg L5: 计算曼哈顿距离得到S_raw(i,j)
    s_raw = manhattan_distances(theta_pca, theta_pca)

    # Alg L6: 导出 0-1 屏蔽矩阵 S_p 阈值为 μ_{S_raw}（仅统计 i!=j）
    if client_number > 1:
        mu_s_raw = (s_raw.sum() - np.trace(s_raw)) / (client_number * (client_number - 1))
    else:
        mu_s_raw = 0.0
    structure_adj_matrix = (s_raw >= mu_s_raw).astype(np.float32)
    np.fill_diagonal(structure_adj_matrix, 0.0)

    # Alg L7: 生成随即图 G_r（具体拓扑在 get_proxy_data，见该函数行注释）
    random_graph = get_proxy_data(args.num_node_features)

    # Alg L8: 计算 B_n = AVG(M_n(G_r; Θ_n)) — 对随机图节点维求均值得到每个客户端的行为向量
    fun_output = [torch.mean(client_model[i](random_graph), dim=0).tolist() for i in range(client_number)]
    fun_similarity_distance = [[0 for i in range(client_number)] for _ in range(client_number)]
    for i in range(client_number):
        for j in range(client_number):
            # B(i,j) 用输出向量余弦相关（1 - cosine_distance）
            fun_similarity_distance[i][j] = 1 - cosine(fun_output[i], fun_output[j])
    
    # Alg L9: 归一化行为相似矩阵 S_b(i,j) — exp(ξ·B) 再按行归一（ξ 此处取 10）
    fun_similarity_distance = [[math.exp(element * 10) for element in row] for row in fun_similarity_distance]
    fun_similarity_distance = [[element / sum(row) for element in row] for row in fun_similarity_distance]

    # Alg L10: S_s = S_p × S_b — 相似矩阵相乘用于构建图的加权邻接矩阵
    adj_matrix = torch.matmul(torch.tensor(structure_adj_matrix, dtype=torch.float32), torch.tensor(fun_similarity_distance, dtype=torch.float32))

    # Alg L11 客户端结构图节点特征 Θ
    struct_graph_feature = []
    for i in range(client_number):
        for name, param in client_model[i].named_parameters():
            if name=="conv2.lin.weight":
                struct_graph_feature.append(param.data.view(-1).tolist())
    struct_graph_feature = torch.tensor(struct_graph_feature, requires_grad=True)
    # Alg L11 客户端结构图边索引和边权重
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
    edge_weight = adj_matrix[adj_matrix.nonzero(as_tuple=True)]

    # Alg L11 构建客户端结构图数据
    data = Data(x=struct_graph_feature, edge_index=edge_index, edge_attr=edge_weight)
    # Alg L11: 结构图生成
    # Alg L11: GAE_encoder(Θ, S_s) → Z；KMeans(Z, ⌊√N⌋) → C
    model = GAE(STRGCNNet(struct_graph_feature.size(1), math.floor(math.sqrt(client_number))))
    z = model.encode(data)
    n_clusters = math.floor(math.sqrt(client_number))
    # Alg L11: KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(z.detach().numpy())
    labels = kmeans.labels_

    # 调用可视化函数
    # visualize_clusters(z, labels, method='pca')  # 或者使用'tsne'

    # 结束计时
    end_time = time.time()
    # 计算操作耗时
    elapsed_time = end_time - start_time
    # print(elapsed_time)
    #下一步工作：测试两种相似度效果，测试以距离得到ratio和以注意力机制得到ratio效果
    # return client_att_dataset_ratio

    # RETURN：S_s（adj_matrix）与聚类标签 C，供 learning 中 Alg L143–147 个性化聚合使用；Alg L149 M_global 在聚合阶段
    return adj_matrix, labels


def get_proxy_data(n_feat):
    """Algorithm 1 Alg L7 的子步骤：随机图 G_r（论文为 N 个簇、每簇 100 节点、p_in=0.1、p_out=0.01；本仓库参数为 5 区、p_out=0）。"""
    import networkx as nx
    num_graphs, num_nodes = 5, 100  # 随机图的分区数，每个分区的节点数
    # 将NetworkX图转换为PyTorch Geometric数据格式。其中每个分区包含num_nodes个节点，分区内的连接概率为0.1，分区间的连接概率为0，随机种子为self.args.seed
    data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=1234))
    # 为生成的图数据添加节点特征，特征服从均值为0、标准差为1的正态分布，特征维度为n_feat。
    data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
    return data


def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    """NetworkX → PyG Data；供 Algorithm 1 Alg L138 生成 G_r 时调用（非伪代码单独一步）。"""
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G # 检查并转换图为有向图（如果原图是无向图）

    # 创建节点映射，并初始化边索引矩阵：
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    # 填充边索引矩阵：
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    # 初始化数据存储的字典：
    from collections import defaultdict
    data = defaultdict(list)

    # 提取节点属性和边属性（如果存在）：
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    # 处理节点数据并存储：
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    # 处理边数据并存储：
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    # 处理图的全局属性：
    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    # 将数据转换为张量格式：
    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    # 设置边索引：
    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    # 合并节点属性（如果指定）：
    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    # 合并边属性（如果指定）：
    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    # 设置节点数量（如果未设置）：
    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    # 返回转换后的数据：
    return data