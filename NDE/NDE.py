import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt


def pLog2p(p_i, eps = 1e-10):
    if p_i < eps:
        p_i = 1.0
    return p_i * np.log2(p_i)

def Log2p(p_i, eps = 1e-10):
    if p_i < eps:
        p_i = 1.0
    return np.log2(p_i)

def NormalizeDegree(adj,node_list):
    adj_2 = adj

    degree_vec = np.sum(adj, axis=0) + 1

    degree2_mtrx_hat = np.diag(np.array(np.power(np.sum(adj_2, axis=0), -0.5)).reshape(-1))

    nor_adj_2 = degree2_mtrx_hat.dot(adj_2).dot(degree2_mtrx_hat)
    nor_degree = np.array(np.dot(degree_vec, nor_adj_2)).squeeze()
    degree = {}
    for index, d in enumerate(nor_degree):
        degree[node_list[index]] = d
    return degree,nor_adj_2

def structure_entropy(adj,node_list):
    degree,nor_adj = NormalizeDegree(adj,node_list)

    G = nx.from_numpy_matrix(nor_adj)
    parts = community.best_partition(G,weight='weight')
    #edges_num = nx.number_of_edges(G)

    commun_nodes = {}  # each community: {nodes}
    for com in set(parts.values()):
        nodes = [node for node in parts.keys() if parts[node] == com]
        commun_nodes[com] = nodes

    commun_degree = {}  # each community : {sum of nodes degree in this com}
    inter_edge = {}     # each community : {intersecting edges num in this com}
    prob = {}
    for commun,nodes_index in commun_nodes.items():
        current_degree = [degree[node_list[node_index]] for node_index in nodes_index]
        commun_degree[commun] = sum(current_degree)
        inter_edge_degree = []
        for nodepairs in list(nx.edge_boundary(G, nodes_index)):
            node_in_idx = nodepairs[0]
            node_out_idx = nodepairs[1]
            weight = nor_adj[node_in_idx,node_out_idx]
            inter_edge_nor = degree[node_list[node_out_idx]] * weight
            inter_edge_degree.append(inter_edge_nor)
        inter_edge[commun] = sum(inter_edge_degree)

        unit_prob = [degree[node_list[node_idx]]/commun_degree[commun] for node_idx in nodes_index]
        plogp_prob = [pLog2p(p, eps=1e-10) for p in unit_prob]
        prob[commun] = sum(plogp_prob)

    count_degree = sum(commun_degree.values())

    inner_entropy = 0
    inter_entropy = 0
    for commun in commun_nodes.keys():
        inner_entropy += - (commun_degree[commun]/count_degree) * prob[commun]
        inter_entropy += - (inter_edge[commun]/count_degree) * Log2p(commun_degree[commun]/count_degree)

    H_s = inner_entropy + inter_entropy
    return H_s

def assign_prob(H,H_list):
    d_list = [abs(H-h) for h in H_list]
    D = sum(d_list)
    p_list = []
    for d in d_list:
        p = np.log2(2*D/(d+D))
        p_list.append(p)
    return p_list

def align_entropy(adj_list,p_list,node_list):
    degree_list = []

    align_adj = np.zeros((len(node_list),len(node_list)),dtype='float')
    for adj,p in zip(adj_list,p_list):
        degree,nor_adj = NormalizeDegree(adj, node_list)
        degree_list.append(degree)
        align_adj += nor_adj * p

    degree = dict()

    for key in degree_list[0].keys():
        degree[key] = p_list[0]*degree_list[0][key]
        for d,p in zip(degree_list[1:],p_list[1:]):
            degree[key] += p * d[key]
    G = nx.from_numpy_matrix(align_adj)
    parts = community.best_partition(G)

    #edges_num = nx.number_of_edges(G)

    commun_nodes = {}  # each community: {nodes}
    for com in set(parts.values()):
        nodes = [node for node in parts.keys() if parts[node] == com]
        commun_nodes[com] = nodes

    commun_degree = {}  # each community : {sum of nodes degree in this com}
    inter_edge = {}  # each community : {intersecting edges num in this com}
    prob = {}
    for commun, nodes_idx in commun_nodes.items():
        current_degree = [degree[node_list[node_idx]] for node_idx in nodes_idx]
        commun_degree[commun] = sum(current_degree)
        #inter_num = nx.cut_size(G, nodes)
        #inter_edge[commun] = inter_num
        inter_edge_degree = []
        for nodepairs in list(nx.edge_boundary(G, nodes_idx)):
            node_in_idx = nodepairs[0]
            node_out_idx = nodepairs[1]
            weight = align_adj[node_in_idx][node_out_idx]
            inter_edge_nor = degree[node_list[node_out_idx]] * weight
            inter_edge_degree.append(inter_edge_nor)
        inter_edge[commun] = sum(inter_edge_degree)

        unit_prob = [degree[node_list[node_idx]] / commun_degree[commun] for node_idx in nodes_idx]
        plogp_prob = [pLog2p(p, eps=1e-10) for p in unit_prob]
        prob[commun] = sum(plogp_prob)

    count_degree = sum(commun_degree.values())

    inner_entropy = 0
    inter_entropy = 0
    for commun in commun_nodes.keys():
        inner_entropy += - (commun_degree[commun] / count_degree) * prob[commun]
        inter_entropy += - (inter_edge[commun] / count_degree) * Log2p(commun_degree[commun] / count_degree)

    H_s = inner_entropy + inter_entropy
    return H_s

def org_entropy(G):
    degree = dict(G.degree)
    parts = community.best_partition(G)

    edges_num = nx.number_of_edges(G)

    commun_nodes = {}  # each community: {nodes}
    for com in set(parts.values()):
        nodes = [node for node in parts.keys() if parts[node] == com]
        commun_nodes[com] = nodes

    commun_degree = {}  # each community : {sum of nodes degree in this com}
    inter_edge = {}  # each community : {intersecting edges num in this com}
    prob = {}
    for commun, nodes in commun_nodes.items():
        current_degree = [degree[node] for node in nodes]
        commun_degree[commun] = sum(current_degree)
        inter_num = nx.cut_size(G, nodes)
        inter_edge[commun] = inter_num
        unit_prob = [degree[node] / commun_degree[commun] for node in nodes]
        plogp_prob = [pLog2p(p, eps=1e-10) for p in unit_prob]
        prob[commun] = sum(plogp_prob)

    count_degree = sum(commun_degree.values())

    inner_entropy = 0
    inter_entropy = 0
    for commun in commun_nodes.keys():
        inner_entropy += - (commun_degree[commun] / count_degree) * prob[commun]
        inter_entropy += - (inter_edge[commun] / (2 * edges_num)) * Log2p(commun_degree[commun] / count_degree)

    H_s = inner_entropy + inter_entropy
    return H_s

if __name__ == "__main__":

    with open('./data/cora/cora.cites') as f:
        node_set = set()
        edge_list = []
        for line in f.readlines():
            cited,citing = eval(line.split('\t')[0]),eval(line.split('\t')[1])
            edge_list.append([cited,citing])
            node_set.update([cited,citing])

    node_list = list(node_set)
    adj_mtrx = np.eye(len(node_list),dtype='float')
    for edge in edge_list:
        adj_mtrx[node_list.index(edge[0]),node_list.index(edge[1])] = 1
        adj_mtrx[node_list.index(edge[1]), node_list.index(edge[0])] = 1

    G = nx.Graph()

    G.add_edges_from(edge_list)

    H_s1 = structure_entropy(adj_mtrx,node_list)
    adj_2 = adj_mtrx.dot(adj_mtrx)
    H_s2 = structure_entropy(adj_2, node_list)

    adj_3 = adj_2.dot(adj_mtrx)
    H_s3 = structure_entropy(adj_3, node_list)

    p_list = assign_prob(H_s1,[H_s1,H_s2,H_s3])

    H_s = align_entropy([adj_mtrx,adj_2,adj_3],p_list,node_list)
    print('Graph entopy of cora is:',H_s)

    N = len(adj_mtrx)
    n = (np.log(N * N) + H_s) / 0.24
    print('Optimal node representation dimension is:',n)
