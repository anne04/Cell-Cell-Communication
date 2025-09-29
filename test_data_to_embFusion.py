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
    X_protein_embedding: dict()
):
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  ccc_pairs = ['from_cell', 'to_cell', 'from_gene_node', 'to_gene_node', 'ligand_gene', 'rec_gene'] 
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    dataset = []
    # each sample has [sender set, receiver set, score]
    print('len ccc pairs: %d'%len(ccc_pairs))
    record_index = []
    for i in range (0, len(ccc_pairs)):
        #print(i)
        sender_cell_barcode = ccc_pairs['from_cell'][i]
        rcv_cell_barcode = ccc_pairs['to_cell'][i]
        if sender_cell_barcode  == rcv_cell_barcode:
            continue # for now, skipping autocrine signals
            
        ligand_gene = ccc_pairs['ligand_gene'][i]
        rec_gene = ccc_pairs['rec_gene'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and \
            rec_gene in gene_node_list_per_spot[rcv_cell_barcode] and \
            ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
            
            ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
            rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
            
            sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
            rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
            #score = ccc_pairs['attention_score'][i]
            dataset.append([sender_set, rcvr_set, -1, ligand_gene, rec_gene])
            record_index.append(i)

    print('len dataset: %d'%len(dataset))
    return dataset, record_index



def get_cellEmb_geneEmb_pairs(
    cell_vs_index: dict(),
    barcode_info_gene: list(),
    X_embedding = np.array,
    X_gene_embedding = np.array,
    X_protein_embedding = np.array
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
            cell_vs_gene_emb[cell_barcode][gene_index] = [np.zeros((1, 512)), X_gene_embedding[gene_index], X_protein_embedding[gene_name]]
            #X_embedding[cell_index_cellnest]
        else:
            not_found[gene_name] = 1
            
    return cell_vs_gene_emb


    

if __name__ == "__main__":
    ###########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_lrbind_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/model_LRbind_LUAD_lymph_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome/model_LRbind_LUAD_lymph_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_r1_LYMPH_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/LUAD_TD1_manualDB/LUAD_TD1_manualDB_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_barcode_info', help='Name of the dataset')    
    parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir/model_LRbind_LUAD_lymph_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_r1_LUAD_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    args = parser.parse_args()

    #######################################################

    lymph_start = 6365547
    ccc_pairs = pd.read_csv(args.lr_lrbind_csv_path, sep=",")
    # ccc_pairs = ['from_cell', 'to_cell', 'from_gene_node', 'to_gene_node', 'ligand_gene', 'rec_gene']                            



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
    #    X_embedding = pickle.load(fp) 
    X_embedding = np.load(args.cell_emb_cellnest_path)
    

    with gzip.open(args.gene_emb_path, 'rb') as fp:   
        X_gene_embedding = pickle.load(fp)

    #X_g = 
    
    with gzip.open(args.protein_emb_path, 'rb') as fp:  
        X_protein_embedding = pickle.load(fp)

    #X_p = 
    
    cell_vs_index = dict()
    for i in range(0, len(barcode_info_cellnest)):
        cell_vs_index[barcode_info_cellnest[i][0]] = i

    
    cell_vs_gene_emb = get_cellEmb_geneEmb_pairs(cell_vs_index, barcode_info_gene, X_embedding, X_gene_embedding, X_protein_embedding)
    
    dataset, record_index = get_dataset(ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding)
    print(len(dataset))
    # save it


    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    with gzip.open('database/'+'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_tanh'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    	pickle.dump([dataset, record_index], fp)
