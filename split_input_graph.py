import numpy as np
import pickle
from collections import defaultdict
import gzip
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='input_graph/' , help='The path to the directory having input graph') 
parser.add_argument( '--data_name', type=str, default='Visium_HD_Human_Colon_Cancer_square_002um_outputs', help='The name of dataset')
parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
parser.add_argument( '--total_subgraphs', type=int, default=10, help='starting index of ligand')
args = parser.parse_args()


if args.data_path == 'input_graph/':
    args.data_path = args.data_path + args.data_name + '/'

if args.metadata_to == 'metadata/':
    args.metadata_to = args.metadata_to + args.data_name + '/'

fp = gzip.open(args.data_path + args.data_name + '_adjacency_records', 'rb')
row_col, edge_weight, lig_rec, total_num_cell = pickle.load(fp)

dict_cell_edge = defaultdict(list) # key = node. values = incoming edges
dict_cell_neighbors = defaultdict(list) # key = node. value = nodes corresponding to incoming edges/neighbors
nodes_active = dict()
for i in range(0, len(row_col)): 
    dict_cell_edge[row_col[i][1]].append(i) # index of the edges
    dict_cell_neighbors[row_col[i][1]].append(row_col[i][0]) # neighbor id
    nodes_active[row_col[i][1]] = '' # to 
    nodes_active[row_col[i][0]] = '' # from


datapoint_size = len(nodes_active.keys())

for i in range (0, datapoint_size):
    neighbor_list = dict_cell_neighbors[i]
    neighbor_list = list(set(neighbor_list))
    dict_cell_neighbors[i] = neighbor_list


fp = gzip.open(args.metadata_to + args.data_name+'_'+'node_id_sorted_xy', 'rb')
node_id_sorted_xy = pickle.load(fp)

node_id_sorted_xy_temp = []
for i in range(0, len(node_id_sorted_xy)):
    if node_id_sorted_xy[i][0] in nodes_active: # skip those which are not in our ROI
        node_id_sorted_xy_temp.append(node_id_sorted_xy[i])

node_id_sorted_xy = node_id_sorted_xy_temp

##################################################################################################################
# one hot vector used as node feature vector
X = np.eye(datapoint_size, datapoint_size)
np.random.shuffle(X)
X_data = X # node feature vector
num_feature = X_data.shape[0]

# split it into N set of edges

total_subgraphs = args.total_subgraphs

edge_list = []
graph_bag = []
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
    #print('graph id %d, node %d to %d'%(set_id,start_index[indx],start_index[indx+1]))
    set1_nodes = []
    set1_edges_index = []
    node_limit_set1 = start_index[indx+1]
    set1_direct_edges = []
    
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
    
    #print('len of set1_edges_index %d'%len(set1_edges_index))
    #if len(set1_edges_index)==0:
    #    break
        
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


    #print('new id: %d'%new_id)
    set1_edges = []
    for i in set1_direct_edges:  #set1_edges_index:
        set1_edges.append([[id_map_old_new[set_id][row_col[i][0]], id_map_old_new[set_id][row_col[i][1]]], edge_weight[i]])
        #set1_edges.append([row_col[i], edge_weight[i]])
        
    edge_list.append(set1_edges)
    
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

    print("number of nodes %d, number of edges %d"%(num_cell, len(row_col_temp)))
    graph_bag.append([X_data, row_col_temp, edge_weight_temp])
    gc.collect()
    

with gzip.open(args.data_path+args.data_name+'_graph_bag', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump([graph_bag, datapoint_size], fp)


