import NewEntropy_v6
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from opt_base import arg_parse,train,test
from util import load_data, separate_data
from models.graphcnn_opt import GraphCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import community
import time

def optimize_dimension(args):
    print('start')
    
    G, num_classes = load_data(args.dataset, args.degree_as_tag)
    
    dimension = []
    for graph in G:
        g = nx.from_edgelist(graph.g.edges())
        adj = nx.to_numpy_matrix(g)
        node_list = list(g.nodes)
        H_s1 = NewEntropy_v6.structure_entropy(adj, node_list)

        adj_2 = adj.dot(adj)
        H_s2 = NewEntropy_v6.structure_entropy(adj_2, node_list)

        adj_3 = adj_2.dot(adj)
        H_s3 = NewEntropy_v6.structure_entropy(adj_3, node_list)

        p_list = NewEntropy_v6.assign_prob(H_s1, [H_s1, H_s2, H_s3])
        H_s =  NewEntropy_v6.align_entropy([adj,adj_2,adj_3],p_list,node_list)

        N = len(adj)
        n = (np.log(N * N) + H_s) / 0.24
        dimension.append(n)
    
    
    #find two dimension center
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(np.array(dimension).reshape(-1, 1))
    centers = kmeans.cluster_centers_
    #twok_1 = int(list(centers.flatten())[0])
    #twok_2 = int(list(centers.flatten())[1])

    centers_list = [int(center) for center in list(centers.flatten())]

    print(centers_list)
    
    exit()
    
    
    
    #G1, G2 = [], []
    G_all = [[]]* args.centers
    
    for label, graph in zip(kmeans.labels_, G):
        #if label == 0:
            #G1.append(graph)
        #else:
            #G2.append(graph)
        G_all[label].append(graph)
    #return num_classes,onek_1,twok_1,twok_2,G1,G2
    
    
    
    return num_classes, G_all, centers_list, onek_1









def task_val(args,num_classes,G,dims,dim0):
    device = args.device
    all_vals = []
    duration = 0
    for fold_idx in range(10):
        print('fold_idx:', fold_idx)
        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        #train_G1, test_G1 = separate_data(G1, args.seed, fold_idx)
        #train_G2, test_G2 = separate_data(G2, args.seed, fold_idx)
        
        train_G = [] 
        test_G = [] 
        for Gi in G:
            train_g, test_g = separate_data(Gi, args.seed, fold_idx)
            train_G.append(train_g)
            test_G.append(test_g)
            
        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_G[1][0].node_features.shape[1], dim0, dims,
                     num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                     args.neighbor_pooling_type, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        val_accs = []
        total_test = sum([len(test) for test in test_G])
        for epoch in range(1, args.epochs + 1):
            scheduler.step()
            test_correct = 0
            
            for road, (train_graphs, test_graphs)  in enumerate(zip(train_G, test_G)):
                avg_loss = train(args, model, device, train_graphs, optimizer, epoch, road)
                test_correct += test(args, model, device, train_graphs, test_graphs, epoch, road)

                if not args.filename == "":
                    with open(args.filename, 'w') as f:
                        #f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                        f.write("\n")
                print("")
                print(model.eps)
                      
            val_accs.append(test_correct/total_test)
                              
        all_vals.append(np.array(val_accs))
      
    all_vals = np.vstack(all_vals)
    mean_vals = np.mean(all_vals, axis=0)
    best_epoch = np.argmax(mean_vals)
    print(mean_vals)
    print(np.max(mean_vals))
    print(best_epoch)
    print(all_vals[:, best_epoch])
    print(np.std(all_vals[:, best_epoch]))
    with open(args.dataset + 'acc_opt.txt','a+') as f:
        f.write(str(args.centers) + '\t' + 'std:' + str(np.std(all_vals[:, best_epoch])) + '\t' + 'acc:' + str(np.max(mean_vals)) + '\t' + str(best_epoch) + '\t')
       
    return all_vals

def main():
    args = arg_parse()
    torch.manual_seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        #optimize_dimension(args)
    
    #num_classes, dim1_1, dim2_1, dim2_2, G1, G2 = optimize_dimension(args)
    
    num_classes,G,dims,dim0 = optimize_dimension(args)
    print(dims,dim0)
    exit()
    
    acc = task_val(args,num_classes,G,dims,dim0)
    
    

if __name__ == "__main__":
    main()