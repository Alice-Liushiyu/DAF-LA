import torch
import torch.nn.functional as F
import numpy as np

# 假设这些是你的输入数据
from sklearn.decomposition import PCA

metabolite_similarity = np.loadtxt('D:\yjs\DAF-LA\data/mn.txt', dtype=float,delimiter='\t')
disease_similarity = np.loadtxt('D:\yjs\DAF-LA\data\data/dn.txt', dtype=float, delimiter='\t')
adjacency_matrix = np.loadtxt('D:\yjs\DAF-LA\data\dm.txt', dtype=float)


# 转换为torch tensor
metabolite_similarity = torch.tensor(metabolite_similarity, dtype=torch.float32)
disease_similarity = torch.tensor(disease_similarity, dtype=torch.float32)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
#

import torch

import numpy as np

base_k=20

import torch
import torch.nn.functional as F
import networkx as nx  # 用于计算局部聚集系数


def compute_clustering_coefficient(edge_index, num_nodes):
    """
    计算每个节点的局部聚集系数
    :param edge_index: 边列表 (2, num_edges)
    :param num_nodes: 节点总数
    :return: 每个节点的局部聚集系数向量 (num_nodes,)
    """
    # 使用 NetworkX 构建图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())  # 将边添加到图中

    # 计算每个节点的局部聚集系数
    clustering_coeffs = nx.clustering(G)

    # 转换为tensor并处理没有聚集系数的节点
    coeff_tensor = torch.tensor([clustering_coeffs.get(i, 0.0) for i in range(num_nodes)])  # 没有聚集系数的节点默认为 0.0
    return coeff_tensor


def construct_dynamic_subgraph_with_clustering(similarity_matrix, base_k=5, edge_index=None):
    """
    根据节点的局部聚集系数动态调整k值构建子图
    :param similarity_matrix: 节点相似性矩阵 (num_nodes, num_nodes)
    :param base_k: 基础k值，随着局部聚集系数动态调整
    :param edge_index: 原始图的边列表，用于计算聚集系数
    :return: 子图的边列表 (每个节点与其最相似的动态k个节点的连接)
    """
    num_nodes = similarity_matrix.shape[0]
    edge_index_list = []

    # 如果传入的edge_index为空，使用空图计算局部聚集系数
    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).t()

    # 计算每个节点的局部聚集系数
    clustering_coeffs = compute_clustering_coefficient(edge_index, num_nodes)

    for i in range(num_nodes):
        # 获取节点的局部聚集系数
        coeff = clustering_coeffs[i]

        # 根据局部聚集系数动态调整k值，聚集系数高时减少k值，稀疏时增加k值
        dynamic_k = max(1, int(base_k / (1 + coeff)))  # 当局部稠密时减少k值

        # 选择最相似的 dynamic_k 个邻居
        sim_scores = similarity_matrix[i]
        neighbors = torch.topk(sim_scores, k=dynamic_k, largest=True).indices

        for neighbor in neighbors:
            edge_index_list.append((i, neighbor.item()))  # 构建边（i, neighbor）

    return torch.tensor(edge_index_list, dtype=torch.long).t()


# 示例：为代谢物相似网络构建动态子图 (1435, 1435)
metabolite_edge_index = construct_dynamic_subgraph_with_clustering(metabolite_similarity, base_k=base_k)

# 示例：为疾病相似网络构建动态子图 (177, 177)
disease_edge_index = construct_dynamic_subgraph_with_clustering(disease_similarity, base_k=base_k)

print(metabolite_edge_index.shape)  # 输出 (2, num_edges)
print(disease_edge_index.shape)  # 输出 (2, num_edges)


def edge_aggregate_mean(node_features, edge_index):
    """
    在子图上进行边缘聚合，计算每个节点的特征（平均池化）
    :param node_features: 初始节点特征
    :param edge_index: 子图边列表
    :return: 聚合后的节点特征
    """
    num_nodes = node_features.shape[0]
    aggregated_features = torch.zeros_like(node_features)

    # 记录每个目标节点的源节点数量
    counts = torch.zeros(num_nodes)

    for edge in edge_index.t():
        src, dst = edge
        aggregated_features[dst] += node_features[src]
        counts[dst] += 1

    # 防止除以零，保持原特征
    counts[counts == 0] = 1
    aggregated_features /= counts.unsqueeze(1)

    return aggregated_features


# 代谢物初始特征 (可以直接使用相似度矩阵或其他特征)
metabolite_initial_features = F.relu(metabolite_similarity)

# 聚合代谢物的特征
metabolite_features = edge_aggregate_mean(metabolite_initial_features, metabolite_edge_index)

# 疾病初始特征
disease_initial_features = F.relu(disease_similarity)

# 聚合疾病的特征
disease_features = edge_aggregate_mean(disease_initial_features, disease_edge_index)

print(metabolite_features.shape)  # (1435, 1435) -> 聚合后的代谢物特征
print(disease_features.shape)  # (177, 177)   -> 聚合后的疾病特征

# 使用 PCA 进行降维到 128 维
def reduce_dimensions_with_pca(features, n_components=128):
    pca = PCA(n_components=n_components,random_state=40)
    # 对特征进行降维
    reduced_features = pca.fit_transform(features.detach().numpy())  # 将张量转换为 NumPy 数组
    return torch.tensor(reduced_features, dtype=torch.float32)

# 直接将负数变成正数，取绝对值
def make_all_positive(features):
    return torch.abs(features)  # 取每个元素的绝对值

# 降维后的代谢物特征
metabolite_features_reduced = reduce_dimensions_with_pca(metabolite_features)
metabolite_features_positive = make_all_positive(metabolite_features_reduced)

# 降维后的疾病特征
disease_features_reduced = reduce_dimensions_with_pca(disease_features)
disease_features_positive = make_all_positive(disease_features_reduced)


print(metabolite_features_reduced.shape)  # (1435, 128)
print(disease_features_reduced.shape)     # (177, 128)

print("Metabolite Features Shape:", metabolite_features_reduced.shape)  # 输出: (1435, 128)
metabolite_features= metabolite_features_reduced.detach().numpy()
np.savetxt(f"D:\yjs\DAF-LA\data/meta_features_{base_k}.txt", metabolite_features ,delimiter='\t')
print("Disease Features Shape:", disease_features_reduced.shape)        # 输出: (177, 128)
disease_features= disease_features_reduced.detach().numpy()
np.savetxt(f"D:\yjs\DAF-LA\data/dis_features_{base_k}.txt", disease_features, delimiter='\t')