import pandas as pd
from collections import defaultdict
import pickle
import argparse
import gzip
import numpy as np
import gc

def get_dataset(
    ccc_pairs: pd.DataFrame,
    cell_vs_gene_emb: defaultdict(dict),
    gene_node_list_per_spot: defaultdict(dict),
    X_protein_embedding: dict(),
    threshold_score: int = 0.7,
    dataset = []
):
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  columns are ['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id', 'attention_score']
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    # each sample has [sender set, receiver set, score]  
    dataset = []      
    for i in range (0, len(ccc_pairs)):
        print("%d/%d - found %d"%(i,len(ccc_pairs),len(dataset)), end='\r')
        sender_cell_barcode = str(ccc_pairs['from_cell'][i])
        rcv_cell_barcode = str(ccc_pairs['to_cell'][i])
        if sender_cell_barcode  == rcv_cell_barcode:
            continue # for now, skipping autocrine signals
            
        ligand_gene = ccc_pairs['ligand'][i]
        rec_gene = ccc_pairs['receptor'][i]
        #sender_cell_index = ccc_pairs['from_id'][i]
        #rcvr_cell_index = ccc_pairs['to_id'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and \
            rec_gene in gene_node_list_per_spot[rcv_cell_barcode] and \
            ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
            
            ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
            rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
            
            sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
            rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
            score = ccc_pairs['attention_score'][i]
            print(threshold_score)
            if score < threshold_score:
                score = 0
            else:
                score = 1
                
            dataset.append([sender_set, rcvr_set, score, ligand_gene, rec_gene])

    print('\nlen dataset: %d'%len(dataset))
    return dataset

def get_dataset_from_allCCC(
    ccc_pairs: pd.DataFrame,
    all_ccc_pairs: pd.DataFrame,
    cell_vs_gene_emb: defaultdict(dict),
    gene_node_list_per_spot: defaultdict(dict),
    X_protein_embedding: dict(),
    threshold_score: int = 0.7,
    dataset = [],
    rem_pair = ""
):
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  columns are ['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id', 'attention_score']
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    # each sample has [sender set, receiver set, score]  
    dataset = []   
    found_edge = dict()   
    for i in range (0, len(ccc_pairs)):
        print("%d/%d - found %d"%(i,len(ccc_pairs),len(dataset)), end='\r')
        sender_cell_barcode = str(ccc_pairs['from_cell'][i])
        rcv_cell_barcode = str(ccc_pairs['to_cell'][i])
        if sender_cell_barcode  == rcv_cell_barcode:
            continue # for now, skipping autocrine signals
            
        ligand_gene = ccc_pairs['ligand'][i]
        rec_gene = ccc_pairs['receptor'][i]
        #sender_cell_index = ccc_pairs['from_id'][i]
        #rcvr_cell_index = ccc_pairs['to_id'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene + '_to_' + rec_gene in rem_pair:
            continue

        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and \
            rec_gene in gene_node_list_per_spot[rcv_cell_barcode] and \
            ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
            

            ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
            rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
            
            sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
            rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
            score = 1
            dataset.append([sender_set, rcvr_set, score, ligand_gene, rec_gene])
            found_edge[sender_cell_barcode + '-' + rcv_cell_barcode + '-' + ligand_gene + '-' + rec_gene] = 1
    print('\nlen poz dataset: %d'%len(dataset))

    for i in range (0, len(all_ccc_pairs)):
        print("%d/%d - found %d"%(i,len(all_ccc_pairs),len(dataset)), end='\r')
        sender_cell_barcode = str(all_ccc_pairs['from_cell'][i])
        rcv_cell_barcode = str(all_ccc_pairs['to_cell'][i])
        if sender_cell_barcode  == rcv_cell_barcode:
            continue # for now, skipping autocrine signals
            
        ligand_gene = all_ccc_pairs['ligand'][i]
        rec_gene = all_ccc_pairs['receptor'][i]
        #sender_cell_index = ccc_pairs['from_id'][i]
        #rcvr_cell_index = ccc_pairs['to_id'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene + '_to_' + rec_gene in rem_pair:
            continue


        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and \
            rec_gene in gene_node_list_per_spot[rcv_cell_barcode] and \
            ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
            
            ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
            rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
            
            sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
            rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
            key_check = sender_cell_barcode + '-' + rcv_cell_barcode + '-' + ligand_gene + '-' + rec_gene
            if key_check not in found_edge:
                score = 0
                dataset.append([sender_set, rcvr_set, score, ligand_gene, rec_gene])
            
    print('\nlen poz+neg dataset: %d'%len(dataset))


    return dataset

def get_negative_dataset(
    ccc_pairs: pd.DataFrame,
    cell_vs_gene_emb: defaultdict(dict),
    gene_node_list_per_spot: defaultdict(dict),
    X_protein_embedding : dict(),
    dataset:list(),
    flag = 'inter'
):
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  columns are ['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id', 'attention_score']
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    # each sample has [sender set, receiver set, score]
    if dataset == None:
        dataset = []

    initial_len = len(dataset)
    protein_emb_notfound = 0
    for i in range (0, len(ccc_pairs)):
        print("%d/%d - found %d"%(i,len(ccc_pairs),len(dataset)-initial_len), end='\r')
        sender_cell_barcode = str(ccc_pairs['from_cell'][i])
        rcv_cell_barcode = str(ccc_pairs['to_cell'][i])
        if flag == 'inter' and sender_cell_barcode  == rcv_cell_barcode:
            continue 
            
        ligand_gene = ccc_pairs['ligand_gene'][i]
        rec_gene = ccc_pairs['rec_gene'][i]
        #sender_cell_index = ccc_pairs['from_id'][i]
        #rcvr_cell_index = ccc_pairs['to_id'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and rec_gene in gene_node_list_per_spot[rcv_cell_barcode]: 
            if ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
                ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
                rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
                
                sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
                rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
                score = 0 #ccc_pairs['attention_score'][i]
                dataset.append([sender_set, rcvr_set, score, ligand_gene, rec_gene])
            else:
                protein_emb_notfound = protein_emb_notfound + 1

    
            
    print('\nlen dataset: %d, protein emb not found %d'%(len(dataset), protein_emb_notfound))
    return dataset




def get_cellEmb_geneEmb_pairs(
    cell_vs_index: dict(),
    barcode_info_gene: list(),
    X_embedding,
    X_gene_embedding: np.array,
    X_protein_embedding: np.array
) -> defaultdict(dict):
    """

    Parameters:
    cell_vs_index: dictionary with key = cell_barcode, value = index of that cell 
    barcode_info_gene: list of [cell's barcode, cell's X, cell's Y, -1, gene_node_index, gene_name]
    X = 2D np.array having row = cell index, column = feature dimension
    X_g = 2D np.array having row = gene node index, column = feature dimension
    """
    
    cell_vs_gene_emb = defaultdict(dict)
    not_found = dict()
    for i in range (0, len(barcode_info_gene)):
        cell_index = i
        cell_barcode = barcode_info_gene[i][0]
        gene_index = barcode_info_gene[i][4]
            
        cell_index_cellnest = cell_vs_index[cell_barcode]
        gene_name = barcode_info_gene[i][5]
        #if cell_barcode == 'GGCGCTCCTCATCAAT-1':
        #    print(gene_index)  
        if gene_name in X_protein_embedding:         
            cell_vs_gene_emb[cell_barcode][gene_index] = [np.zeros(512), X_gene_embedding[gene_index], X_protein_embedding[gene_name]]
            #X_embedding[cell_index_cellnest]
        else:
            not_found[gene_name] = 1
            
    return cell_vs_gene_emb

def combined_graph(args):
    args.metadata_from = args.metadata_from + args.data_name + '/'
    args.data_from = args.data_from + args.data_name + '/'
    args.embedding_path  = args.embedding_path + args.data_name + '/'        
    ##################### get metadata: barcode_info ###################################
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info = pickle.load(fp) 

    barcode_index = dict()
    for i in range (0, len(barcode_info)):
        barcode_index[barcode_info[i][0]] = i
    
    
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, gene_node_index_active, ligand_active, receptor_active = pickle.load(fp)

    print('total gene node %d'%len(list(gene_node_index_active.keys())))
    

    print('****' + args.data_name + '*********')
    #args.model_name = model_names[data_index]

    ######################### LR database ###############################################
    df = pd.read_csv(args.database_path, sep=",")
    db_gene_nodes = dict()
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        db_gene_nodes[ligand] = '1'
        db_gene_nodes[receptor] = '1'


    ##################################################################
    fp = gzip.open('input_graph/'+args.data_name+'/'+ args.data_name +'_adjacency_gene_records', 'rb')  
    row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(fp)
    ########### Merge the subgraphs ####################
    dict_cell_edge = defaultdict(list) # key = node. values = incoming edges
    dict_cell_neighbors = defaultdict(list) # key = node. value = nodes corresponding to incoming edges/neighbors

    nodes_active = dict()
    for i in range(0, len(row_col_gene)): 
        dict_cell_edge[row_col_gene[i][1]].append(i) # index of the edges (incoming)
        dict_cell_neighbors[row_col_gene[i][1]].append(row_col_gene[i][0]) # neighbor id
        nodes_active[row_col_gene[i][1]] = '' # to 
        nodes_active[row_col_gene[i][0]] = '' # from
    
    datapoint_size = len(nodes_active.keys())
    for i in range (0, datapoint_size):
        neighbor_list = dict_cell_neighbors[i]
        neighbor_list = list(set(neighbor_list))
        dict_cell_neighbors[i] = neighbor_list


    node_id_sorted_path = args.metadata_from + '/'+ args.data_name+'_'+'gene_node_id_sorted_xy'
    fp = gzip.open(node_id_sorted_path, 'rb')
    node_id_sorted_xy = pickle.load(fp)
    node_id_sorted_xy_temp = []
    for i in range(0, len(node_id_sorted_xy)):
        if node_id_sorted_xy[i][0] in nodes_active: # skip those which are not in our ROI
            node_id_sorted_xy_temp.append(node_id_sorted_xy[i])
    
    node_id_sorted_xy = node_id_sorted_xy_temp    
    
    ##################################################################################################################
    # split it into N set of edges    
    total_subgraphs = args.total_subgraphs 
    #edge_list = []
    graph_bag = []
    start_index = []
    id_map_old_new = [] # make an index array, so that existing node ids are mapped to new ids
    id_map_new_old = []
    
    for i in range (0, total_subgraphs+1):
        start_index.append((datapoint_size//total_subgraphs)*i)
        id_map_old_new.append(dict())
        id_map_new_old.append(dict())

    
    ##################################################################################################################
    set_id=-1
    suggraph_vs_gene_emb = defaultdict(list)
    for indx in range (0, len(start_index)-1):
        set_id = set_id + 1
        #print('graph id %d, node %d to %d'%(set_id,start_index[indx],start_index[indx+1]))
        set1_nodes = []
        set1_edges_index = []
        node_limit_set1 = start_index[indx+1]
        set1_direct_edges = []

        for i in range (start_index[indx], node_limit_set1):
            set1_nodes.append(node_id_sorted_xy[i][0])
            suggraph_vs_gene_emb[set_id].append(node_id_sorted_xy[i][0])
            # add it's incoming edges - first hop
            for edge_index in dict_cell_edge[node_id_sorted_xy[i][0]]: 
                set1_edges_index.append(edge_index) # has both row_col_gene and edge_weight
                set1_direct_edges.append(edge_index)
            # add it's neighbor's edges - second hop
            for neighbor in dict_cell_neighbors[node_id_sorted_xy[i][0]]:
                if node_id_sorted_xy[i][0] == neighbor:
                    continue
                for edge_index in dict_cell_edge[neighbor]:
                    set1_edges_index.append(edge_index) # has both row_col_gene and edge_weight
    
        set1_edges_index = list(set(set1_edges_index))

        #print('len of set1_edges_index %d'%len(set1_edges_index))
        #if len(set1_edges_index)==0:
        #    break

        # old to new mapping of the nodes
        # make an index array, so that existing node ids are mapped to new ids
        new_id = 0
        spot_list = []
        for k in set1_edges_index:
            i = row_col_gene[k][0]
            j = row_col_gene[k][1]
            if i not in id_map_old_new[set_id]:
                id_map_old_new[set_id][i] = new_id
                id_map_new_old[set_id][new_id] = i
                spot_list.append(i) #new_id)
                new_id = new_id + 1
    
            if j not in id_map_old_new[set_id]:
                id_map_old_new[set_id][j] = new_id
                id_map_new_old[set_id][new_id] = j
                spot_list.append(j) #new_id)
                new_id = new_id + 1
    
    
        #print('new id: %d'%new_id)
        set1_edges = []
        for i in set1_direct_edges:  #set1_edges_index:
            set1_edges.append([[id_map_old_new[set_id][row_col_gene[i][0]], id_map_old_new[set_id][row_col_gene[i][1]]], edge_weight[i]])
            #set1_edges.append([row_col_gene[i], edge_weight[i]])

        #edge_list.append(set1_edges)
        num_nodes = new_id
        row_col_gene_temp = []
        edge_weight_temp = []
        for i in range (0, len(set1_edges)):
            row_col_gene_temp.append(set1_edges[i][0])
            edge_weight_temp.append(set1_edges[i][1])
    
        print("subgraph %d: number of nodes %d, Total number of edges %d"%(set_id, num_nodes, len(row_col_gene_temp)))



        gc.collect()



    ######## embedding merge from subgraphs ########################################
    X_embedding = np.zeros((total_num_gene_node, 256)) # args.hidden_dimension
    for subgraph_id in range(0, args.total_subgraphs):
        X_embedding_filename =  args.embedding_path + args.model_name + '_r1_Embed_X' + '_subgraph'+str(subgraph_id)
        with gzip.open(X_embedding_filename, 'rb') as fp:  
            X_embedding_sub = pickle.load(fp)

        for old_id in suggraph_vs_gene_emb[subgraph_id]:
            new_id = id_map_old_new[subgraph_id][old_id]
            X_embedding[old_id, :] = X_embedding_sub[new_id, :]

    print('min %g max %g'%(np.min(X_embedding), np.max(X_embedding)))

    return X_embedding
    
    
def get_final_dataset(args, add_negative=0, add_inter=1, add_intra=1, rem_pair = ""):
    with gzip.open(args.barcode_info_cellnest_path, 'rb') as fp:     
        barcode_info_cellnest = pickle.load(fp)
        
    with gzip.open(args.barcode_info_path, 'rb') as fp:     
        barcode_info = pickle.load(fp)

    with gzip.open(args.barcode_info_gene_path, 'rb') as fp: 
        barcode_info_gene, na, na, gene_node_list_per_spot, na, na, na, na, na = pickle.load(fp)

    gene_node_list_per_spot_temp = defaultdict(dict)
    for cell_index in range(0, len(barcode_info)): 
        gene_node_list_per_spot_temp[barcode_info[cell_index][0]] = gene_node_list_per_spot[cell_index]
        
    gene_node_list_per_spot = gene_node_list_per_spot_temp 
    gene_node_list_per_spot_temp = 0
    gc.collect()
    
    #with gzip.open(args.cell_emb_cellnest_path, 'rb') as fp:  
    #   X_embedding = pickle.load(fp) 

    
    if args.total_subgraphs == 1:
        with gzip.open(args.gene_emb_path, 'rb') as fp:  
            X_gene_embedding = pickle.load(fp)

    else:
        X_gene_embedding = combined_graph(args)

    print('min %g max %g'%(np.min(X_gene_embedding), np.max(X_gene_embedding)))
    if args.normalized_gene_emb == 1:
        for i in range (0, X_gene_embedding.shape[0]):
            total_score_per_row = np.sum(X_gene_embedding[i][:])
            X_gene_embedding[i] = X_gene_embedding[i]/total_score_per_row


    with gzip.open(args.protein_emb_path, 'rb') as fp:  
        X_protein_embedding = pickle.load(fp)

    #X_p = 
    
    cell_vs_index = dict()
    for i in range(0, len(barcode_info_cellnest)):
        cell_vs_index[barcode_info_cellnest[i][0]] = i

    
    cell_vs_gene_emb = get_cellEmb_geneEmb_pairs(cell_vs_index, barcode_info_gene, 0, X_gene_embedding, X_protein_embedding)

    ccc_pairs = pd.read_csv(args.lr_cellnest_csv_path, sep=",")
    print(ccc_pairs.columns)
    all_ccc_pairs = pd.read_csv(args.all_lr_cellnest_csv_path, sep=",")
    #dataset = get_dataset(ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding)
    dataset = get_dataset_from_allCCC(ccc_pairs, all_ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding, rem_pair=rem_pair)


    print(len(dataset))
    start_of_negative_pairs = -1

    if add_negative == 1:
        if add_inter == 1:
            start_of_negative_pairs = len(dataset)
            neg_ccc_pairs = pd.read_csv(args.lr_negatome_inter_csv_path, sep=",")
            dataset = get_negative_dataset(neg_ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding, dataset)
            print(len(dataset))

        if add_intra == 1:
            neg_ccc_pairs = pd.read_csv(args.lr_negatome_intra_csv_path, sep=",")
            dataset = get_negative_dataset(neg_ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding, dataset, 'intra')
            print(len(dataset))
    
    return dataset, start_of_negative_pairs



if __name__ == "__main__":

    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST_experimental/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_top20percent.csv', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST_experimental/metadata/V1_Human_Lymph_Node_spatial/V1_Human_Lymph_Node_spatial_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST_experimental/embedding_data/V1_Human_Lymph_Node_spatial/NEST_V1_Human_Lymph_Node_spatial_r1_Embed_X.npy', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_tanh_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_tanh_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR

    parser.add_argument( '--total_subgraphs', type=int, default=1, help='')
    args = parser.parse_args()

    dataset, start_of_negative_pairs = get_final_dataset(args, add_negative=0)

    """

    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_top20percent.csv', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--all_lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_allCCC.csv', help='Name of the dataset') #, required=True)
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_r1_Embed_X.npy', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR

    parser.add_argument( '--total_subgraphs', type=int, default=1, help='')
    args = parser.parse_args()

    dataset = []
    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=0)

    for i in range(0, len(dataset_temp)):
        dataset.append(dataset_temp[i])
        
    
    #dataset = dataset_temp
    print('len of dataset %d, start of negative pairs %d'%(len(dataset), start_of_negative_pairs))


    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_top20percent.csv', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--all_lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_allCCC.csv', help='Name of the dataset') #, required=True)
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_r1_Embed_X.npy', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_r1_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--lr_negatome_intra_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_r1_negatomeLR_nodeInfo_intra.csv.gz',\
                        help='Name of the dataset') #, required=True)
    parser.add_argument( '--lr_negatome_inter_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_r1_negatomeLR_nodeInfo_inter.csv.gz',\
                        help='Name of the dataset') #, required=True) 
    parser.add_argument( '--total_subgraphs', type=int, default=1, help='')
    parser.add_argument( '--normalized_gene_emb', type=int, default=1, help='')
    args = parser.parse_args()

    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=1, add_inter=1, add_intra=1) #, rem_pair='CXCL12_to_CXCR4'
    print('len of dataset %d, start of negative pairs %d'%(len(dataset_temp), start_of_negative_pairs))

    
    dataset = []
    dataset_negatome = []

    for i in range(0, start_of_negative_pairs): 
        dataset.append(dataset_temp[i])
        
    for i in range(start_of_negative_pairs, len(dataset_temp)):
        dataset_negatome.append(dataset_temp[i])
    
    
    print('len of dataset %d'%(len(dataset)))
    print('len of negatome dataset %d'%(len(dataset_negatome)))

        
    
    #dataset = dataset_temp
    #print('len of dataset %d, start of negative pairs %d'%(len(dataset), start_of_negative_pairs))


    """

    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_top20percent.csv', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--all_lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_FFPE_Human_Breast_Cancer_Rep1/CellNEST_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_allCCC.csv', help='Name of the dataset') #, required=True)
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_r1_Embed_X.npy', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR

    parser.add_argument( '--total_subgraphs', type=int, default=16, help='')
    #######################################################################################################
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local', help='The name of dataset') #, required=True) # default='',
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--model_name', type=str, default='model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local', help='Path to grab the metadata')
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')


    parser.add_argument( '--lr_negatome_intra_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_negatomeLR_nodeInfo_intra.csv',\
                        help='Name of the dataset') #, required=True)
    parser.add_argument( '--lr_negatome_inter_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local/model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_local_negatomeLR_nodeInfo_inter.csv',\
                        help='Name of the dataset') #, required=True)  
        
            
                
    args = parser.parse_args()

    dataset = []
    dataset_negatome = []
    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=1, add_inter=1, add_intra=1)
    print('len of dataset %d, start of negative pairs %d'%(len(dataset_temp), start_of_negative_pairs))

    for i in range(0, start_of_negative_pairs): 
        dataset.append(dataset_temp[i])
        
    for i in range(start_of_negative_pairs, len(dataset_temp)):
        dataset_negatome.append(dataset_temp[i])
    
    
    print('len of dataset %d'%(len(dataset)))


    """


    """
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_Prime_Human_Skin_FFPE/CellNEST_Xenium_Prime_Human_Skin_FFPE_manualDB_top20percent.csv', help='Name of the dataset') #, required=True)
    parser.add_argument( '--all_lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_Prime_Human_Skin_FFPE/CellNEST_Xenium_Prime_Human_Skin_FFPE_manualDB_allCCC.csv', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_barcode_info'  , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_r1_Embed_X_subgraphs_combined', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')

    parser.add_argument( '--total_subgraphs', type=int, default=16, help='')
    #######################################################################################################
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir', help='The name of dataset') #, required=True) # default='',
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--model_name', type=str, default='model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir', help='Path to grab the metadata')
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
    
        
            
                
    args = parser.parse_args()
    
    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=0)

    
    for i in range(0, len(dataset_temp)):
        dataset.append(dataset_temp[i])
        
    
    #dataset = dataset_temp
    print('len of dataset %d, start of negative pairs %d'%(len(dataset), start_of_negative_pairs))


    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_Prime_Human_Skin_FFPE/CellNEST_Xenium_Prime_Human_Skin_FFPE_manualDB_top20percent.csv', help='Name of the dataset') #, required=True)
    parser.add_argument( '--all_lr_cellnest_csv_path', type=str, default='../NEST/output/Xenium_Prime_Human_Skin_FFPE/CellNEST_Xenium_Prime_Human_Skin_FFPE_manualDB_allCCC.csv', help='Name of the dataset')
    parser.add_argument( '--lr_inter_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_barcode_info'  , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_barcode_info', help='Name of the dataset')    
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_r1_Embed_X_subgraphs_combined', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')

    parser.add_argument( '--total_subgraphs', type=int, default=16, help='')
    parser.add_argument( '--normalized_gene_emb', type=int, default=1, help='')
    #######################################################################################################
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome', help='The name of dataset') #, required=True) # default='',
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--model_name', type=str, default='model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome', help='Path to grab the metadata')
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')

    parser.add_argument( '--lr_negatome_intra_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_negatomeLR_nodeInfo_intra.csv.gz',\
                        help='Name of the dataset') #, required=True)
    parser.add_argument( '--lr_negatome_inter_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneCorrKNN_bidir_negatome_negatomeLR_nodeInfo_inter.csv.gz',\
                        help='Name of the dataset') #, required=True)  
        
        
            
                
    args = parser.parse_args()
    ############
    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=1, add_inter=1, add_intra=1, rem_pair='TGFB1_to_TGFBR2')
    print('len of dataset %d, start of negative pairs %d'%(len(dataset_temp), start_of_negative_pairs))

    for i in range(0, start_of_negative_pairs): 
        dataset.append(dataset_temp[i])
        
    for i in range(start_of_negative_pairs, len(dataset_temp)):
        dataset_negatome.append(dataset_temp[i])
    
    print('len of dataset %d'%(len(dataset)))
    print('len of negatome dataset %d'%(len(dataset_negatome)))


    ####
    start_of_negative_pairs = len(dataset)
    dataset = dataset + dataset_negatome
    print('len of dataset %d, start of negative pairs %d'%(len(dataset), start_of_negative_pairs))
    with gzip.open('database/'+'LRbind_Xe_br_sk_Negatome'+'_dataset_embFusion.pkl', 'wb') as fp:  
    	pickle.dump([dataset, start_of_negative_pairs], fp)

        
    """

    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_allCCC.csv', help='Name of the dataset') #, required=True)  #allCCC
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/LUAD_TD1_manualDB/LUAD_TD1_manualDB_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome_barcode_info', help='Name of the dataset') 
    parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    parser.add_argument( '--lr_negatome_intra_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_negatomeLR_nodeInfo_intra.csv',\
                        help='Name of the dataset') #, required=True)
    parser.add_argument( '--lr_negatome_inter_csv_path', type=str, \
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_negatomeLR_nodeInfo_inter.csv',\
                        help='Name of the dataset') #, required=True)  
    parser.add_argument( '--total_subgraphs', type=int, default=1, help='')
    args = parser.parse_args()


    dataset_temp, start_of_negative_pairs = get_final_dataset(args, add_negative=1, add_inter=1, add_intra=1)
    
    start_of_negative_pairs = start_of_negative_pairs + len(dataset)
    for i in range(0, len(dataset_temp)):
        dataset.append(dataset_temp[i])
        
    
    #dataset = dataset_temp
    print('len of dataset %d, start of negative pairs %d'%(len(dataset), start_of_negative_pairs))

    with gzip.open('database/'+'LRbind_Xe_br_sk_woNegatome'+'_dataset_embFusion.pkl', 'wb') as fp:  
    	pickle.dump([dataset, start_of_negative_pairs], fp)


    #with gzip.open('database/'+'LRbind_Lymph_Xe_br_sk_LUAD'+'_dataset_embFusion.pkl', 'wb') as fp:  
    #	pickle.dump([dataset, start_of_negative_pairs], fp)

    # save it
    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrLocalKNN_bidir_interNegatome'+'_dataset_embFusion.pkl', 'wb') as fp:  
    #	pickle.dump([dataset, start_of_negative_pairs], fp)

    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrLocalKNN_bidir_wonegatome'+'_dataset_embFusion.pkl', 'wb') as fp:  
    #	pickle.dump([dataset, start_of_negative_pairs], fp)

    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrLocalKNN_bidir_negatome'+'_dataset_embFusion.pkl', 'wb') as fp:  
    #	pickle.dump([dataset, start_of_negative_pairs], fp)
    #with gzip.open('database/'+'LRbind_LUAD_LYMPH_geneCorrLocalKNN_bidir_negatome'+'_dataset_embFusion_top20p.pkl', 'wb') as fp:  
    #	pickle.dump([dataset, start_of_negative_pairs], fp)
        
        
    unique_gene = dict()
    for i in range(0, len(barcode_info_gene)):
        unique_gene[barcode_info_gene[i][5]] = 1
    print(len(unique_gene))
    count = 0
    for gene in unique_gene:
        if gene in X_protein_embedding:
            count = count+1

    print(count)

