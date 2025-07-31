import pandas as pd
from collections import defaultdict
import pickle
import argparse
import gzip
import numpy as np

def get_dataset(
    ccc_pairs: pd.DataFrame,
    cell_vs_gene_emb: defaultdict(dict),
    gene_node_list_per_spot: defaultdict(dict)
) -> list():
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
        print(i)
        sender_cell_barcode = ccc_pairs['from_cell'][i]
        rcv_cell_barcode = ccc_pairs['to_cell'][i]
        ligand_gene = ccc_pairs['ligand'][i]
        rec_gene = ccc_pairs['receptor'][i]
        sender_cell_index = ccc_pairs['from_id'][i]
        rcvr_cell_index = ccc_pairs['to_id'][i]
        # need to find the index of gene nodes in cells
        
        ligand_node_index = gene_node_list_per_spot[sender_cell_index][ligand_gene]
        rec_node_index = gene_node_list_per_spot[rcvr_cell_index][rec_gene]
        
        sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
        rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
        score = ccc_pairs['attention_score']
        dataset.append([sender_set, rcvr_set, score])

    return dataset



def get_cellEmb_geneEmb_pairs(
    cell_vs_index: dict(),
    barcode_info_gene: list(),
    X = np.array,
    X_g = np.array,
    X_p = np.array
) -> defaultdict(dict):
    """

    Parameters:
    cell_vs_index: dictionary with key = cell_barcode, value = index of that cell 
    barcode_info_gene: list of [cell's barcode, cell's X, cell's Y, -1, gene_node_index, gene_name]
    X = 2D np.array having row = cell index, column = feature dimension
    X_g = 2D np.array having row = gene node index, column = feature dimension
    """
    
    cell_vs_gene_emb = defaultdict(dict)
    for item in barcode_info_gene:
        cell_barcode = item[0]
        gene_index = item[4]
        cell_index = cell_vs_index[cell_barcode]
        gene_name = item[5]
        if gene_name in X_p:
            cell_vs_gene_emb[cell_barcode][gene_index] = [X[cell_index], X_g[gene_index], X_p[gene_name]]

    return cell_vs_gene_emb


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--lr_cellnest_csv_path', type=str, default='../NEST/output/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_allCCC.csv', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/LUAD_TD1_manualDB/LUAD_TD1_manualDB_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered/model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_tanh_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    args = parser.parse_args()

    ccc_pairs = pd.read_csv(args.lr_cellnest_csv_path, sep=",")

    with gzip.open(args.barcode_info_cellnest_path, 'rb') as fp:     
        barcode_info = pickle.load(fp)

    with gzip.open(args.barcode_info_gene_path, 'rb') as fp: 
        barcode_info_gene, na, na, gene_node_list_per_spot, na, na, na, na, na = pickle.load(fp)

    
    with gzip.open(args.cell_emb_cellnest_path, 'rb') as fp:  
        X_embedding = pickle.load(fp) 

    

    with gzip.open(args.gene_emb_path, 'rb') as fp:  
        X_gene_embedding = pickle.load(fp)

    X_g = 
    
    with gzip.open(args.protein_emb_path, 'rb') as fp:  
        X_protein_embedding = pickle.load(fp)

    X_p = 
    
    cell_vs_index = dict()
    for i in range(0, len(barcode_info)):
        cell_vs_index[barcode_info[i][0]] = i

    
    cell_vs_gene_emb = get_cellEmb_geneEmb_pairs(cell_vs_index, barcode_info_gene, X_embedding, X_gene_embedding, X_protein_embedding)
    dataset = get_dataset(ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot)

    # save it
    
