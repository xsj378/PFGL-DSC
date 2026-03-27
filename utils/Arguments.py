from pathlib import Path

# 项目根目录下的 save（相对本文件 utils/Arguments.py 为 ../save）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 50  # 4
        self.local_epochs = 3  # 4
        self.lr = 0.01
        self.un_lr = 0.01
        self.momentum = 0.9
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = True
        self.client_number = 20
        self.save_path = str(_PROJECT_ROOT / "save")
        self.unl_batch_size = 1024
        self.unl_epochs = 5
        self.a = 0.8  # 衰减率
        self.e = 10  # 恢复训练轮次
        self.t = 0.10  # 提早-停止阈值
        self.ulclient_index = 0 # 忘却客户端编号
        # self.embed_dim=128 #点的向量维度
        self.dataset = 'CITESEER'
        if self.dataset == 'CORA':
            self.num_node_features = 1433  # 图节点的特征向量长度
            self.num_classes = 7  # 图标签类别个数
        if self.dataset == 'PUBMED':
            self.num_node_features = 500  # 图节点的特征向量长度
            self.num_classes = 3  # 图标签类别个数
        if self.dataset == 'CITESEER':
            self.num_node_features = 3703  # 图节点的特征向量长度
            self.num_classes = 6  # 图标签类别个数
        if self.dataset == 'Computers':
            self.num_node_features = 767  # 图节点的特征向量长度
            self.num_classes = 10  # 图标签类别个数
        # PFGL-DSC、fedavg、fedprox、fedper
        self.aggregation_method = 'PFGL-DSC'
        # --- PFGL-DSC / method.tex 超参数 ---
        # 行为相似度式中 ξ；增大可增强同簇客户端相似度
        self.ste_xi = 5.0
        # 偏好划分阈值 τ（论文式 equ:partitionnodes）：L_i > τ 的节点归入偏好子图 G_n^p
        self.tau_preference = 0.5
        # 忘却损失 L_eucl 中 λ
        self.lambda_eucl = 0.05
        # Algorithm 2 内层循环次数
        self.unlearn_inner_steps = 2
        # True：按损失划分 G_n^p / G_n^u 并仅在 G_n^u 上忘却、G_n^p 上恢复
        # False：沿用按节点比例切分的基线（便于对比实验）
        self.use_preference_split = False
        # 消融实验 ALL、STE、PFU
        self.ablation = 'ALL'
        # 忘却算法 PFGL-DSC、FedRetraining、KD
        self.unlearning_method = 'PFGL-DSC'

def Arg():
    return Arguments()

if __name__ == '__main__':
    print(Arg())
