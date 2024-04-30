import python as np
import pickle
from collections import defaultdict
import gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='input_graph/' , help='The path to the directory having input graph') 
parser.add_argument( '--data_name', type=str, default='Visium_HD_Human_Colon_Cancer_square_002um_outputs', help='The name of dataset')
parser.add_argument( '--slice_count', type=int, default=6, help='starting index of ligand')
args = parser.parse_args()

total_subgraphs = args.slice_count

fp = gzip.open(args.data_path + args.data_name + '_adjacency_records', 'rb')  
row_col, edge_weight, lig_rec, total_num_cell = pickle.load(fp)

datapoint_size = total_num_cell

lig_rec_dict = []
for i in range (0, datapoint_size):
    lig_rec_dict.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []

total_type = np.zeros((2))        
for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        lig_rec_dict[i][j].append(lig_rec[index])  
##################################################################################################################


# split it into N set of edges
dict_cell_edge = defaultdict(list) # key = node. values = incoming edges
dict_cell_neighbors = defaultdict(list) # key = node. value = nodes corresponding to incoming edges/neighbors
for i in range(0, len(row_col)): 
    dict_cell_edge[row_col[i][1]].append(i) # index of the edges
    dict_cell_neighbors[row_col[i][1]].append(row_col[i][0]) # neighbor id

for i in range (0, datapoint_size):
    neighbor_list = dict_cell_neighbors[i]
    neighbor_list = list(set(neighbor_list))
    dict_cell_neighbors[i] = neighbor_list

edge_list = []
start_index = []
id_map_old_new = [] # make an index array, so that existing node ids are mapped to new ids
id_map_new_old = []


for i in range (0, total_subgraphs+1):
    start_index.append((datapoint_size//total_subgraphs)*i)
    id_map_old_new.append(dict())
    id_map_new_old.append(dict())

set_id=-1
for indx in range (0, len(start_index)-1):
    set_id = set_id + 1
    print('start index is %d'%start_index[indx])
    set1_nodes = []
    set1_edges_index = []
    node_limit_set1 = start_index[indx+1]
    set1_direct_edges = []
    print('set has nodes upto: %d'%node_limit_set1)
    for i in range (start_index[indx], node_limit_set1):
        set1_nodes.append(node_id_sorted_xy[i][0])
        # add it's edges - first hop
        for edge_index in dict_cell_edge[node_id_sorted_xy[i][0]]:
            set1_edges_index.append(edge_index) # has both row_col and edge_weight
            set1_direct_edges.append(edge_index)
        # add it's neighbor's edges - second hop
        for neighbor in dict_cell_neighbors[node_id_sorted_xy[i][0]]:
            if node_id_sorted_xy[i][0] == neighbor:
                continue
            for edge_index in dict_cell_edge[neighbor]:
                set1_edges_index.append(edge_index) # has both row_col and edge_weight

    set1_edges_index = list(set(set1_edges_index))
    print('amount of edges in set is: %d'%len(set1_edges_index))

    # old to new mapping of the nodes
    # make an index array, so that existing node ids are mapped to new ids
    new_id = 0
    spot_list = []
    for k in set1_edges_index:
        i = row_col[k][0]
        j = row_col[k][1]
        if i not in id_map_old_new[set_id]:
            id_map_old_new[set_id][i] = new_id
            id_map_new_old[set_id][new_id] = i
            spot_list.append(new_id)
            new_id = new_id + 1

        if j not in id_map_old_new[set_id]:
            id_map_old_new[set_id][j] = new_id
            id_map_new_old[set_id][new_id] = j
            spot_list.append(new_id)
            new_id = new_id + 1


    print('new id: %d'%new_id)
    set1_edges = []
    for i in set1_direct_edges:  #set1_edges_index:
        set1_edges.append([[id_map_old_new[set_id][row_col[i][0]], id_map_old_new[set_id][row_col[i][1]]], edge_weight[i]])
        #set1_edges.append([row_col[i], edge_weight[i]])
        
    edge_list.append(set1_edges)
    '''
    # create new X matrix
    num_cell = new_id
    X_data = np.zeros((num_cell, datapoint_size))
    spot_id = 0
    for spot in spot_list:
        X_data[spot_id] = X[spot,:]
        spot_id = spot_id + 1    
    
    row_col_temp = []
    edge_weight_temp = []
    for i in range (0, len(set1_edges)):
        row_col_temp.append(set1_edges[i][0])
        edge_weight_temp.append(set1_edges[i][1])

    edge_index = torch.tensor(np.array(row_col_temp), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight_temp), dtype=torch.float)
    edge_list.append([X_data, edge_index, edge_attr])
    gc.collect()
    '''




