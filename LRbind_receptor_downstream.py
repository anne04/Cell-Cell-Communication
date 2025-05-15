# Written By 
# Fatema Tuz Zohora


print('package loading')
import numpy as np
import pickle
from scipy import sparse
import numpy as np
import qnorm
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
import gzip
import argparse
import os
import scanpy as sc
import pathway_search_LRbind as pathway


print('user input reading')
#current_dir = 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--data_name', type=str, default='LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir', required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--data_from', type=str, default='../data/LUAD/LUAD_GSM5702473_TD1/' , \
                        help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the \
                        *.mtx file of the gene expression matrix.', required=True) 
    ################# default is set ################################################################
    parser.add_argument( '--data_to', type=str, default='input_graph/', help='Path to save the input graph (to be passed to GAT)')
    parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
    parser.add_argument( '--filter_min_cell', type=int, default=1 , help='Minimum number of cells for gene filtering') 
    parser.add_argument( '--threshold_gene_exp', type=float, default=98, help='Threshold percentile for gene expression. Genes above this percentile are considered active.')
    parser.add_argument( '--tissue_position_file', type=str, default='../data/LUAD/LUAD_GSM5702473_TD1/GSM5702473_TD1_tissue_positions_list.csv', help='If your --data_from argument points to a *.mtx file instead of Space Ranger, then please provide the path to tissue position file.')
    #parser.add_argument( '--num_hops', type=int, default=3 , help='Number of hops for direct connection')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database_no_predictedPPI.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.') 
    parser.add_argument( '--intra_human_ppi_path', type=str, default='database/human_signaling_ppi.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.') 
    parser.add_argument( '--intra_human_tf_target_path', type=str, default='database/human_tf_target.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.') 
    parser.add_argument( '--intra_cutoff', type=float, default=0.3 , help='?') 
    args = parser.parse_args()
    

    if args.data_to == 'input_graph/':
        args.data_to = args.data_to + args.data_name + '/'
    if not os.path.exists(args.data_to):
        os.makedirs(args.data_to)

    if args.metadata_to == 'metadata/':
        args.metadata_to = args.metadata_to + args.data_name + '/'
    if not os.path.exists(args.metadata_to):
        os.makedirs(args.metadata_to)

    ####### get the gene id, cell barcode, cell coordinates ######
    print('input data reading')
    if args.tissue_position_file == 'None': # Data is available in Space Ranger output format
        adata_h5 = sc.read_visium(path=args.data_from, count_file='filtered_feature_bc_matrix.h5')
        print('input data read done')
        gene_count_before = len(list(adata_h5.var_names) )    
        sc.pp.filter_genes(adata_h5, min_cells=args.filter_min_cell)
        gene_count_after = len(list(adata_h5.var_names) )  
        print('Gene filtering done. Number of genes reduced from %d to %d'%(gene_count_before, gene_count_after))
        gene_ids = list(adata_h5.var_names)
        coordinates = adata_h5.obsm['spatial']
        cell_barcode = np.array(adata_h5.obs.index)
        print('Number of barcodes: %d'%cell_barcode.shape[0])
        print('Applying quantile normalization')
        temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  #https://en.wikipedia.org/wiki/Quantile_normalization
        cell_vs_gene = np.transpose(temp)      
    
    else: # Data is not available in Space Ranger output format
        # read the mtx file

       	temp = sc.read_10x_mtx(args.data_from) #
        print(temp)
        print('*.mtx file read done')
        gene_count_before = len(list(temp.var_names))
        sc.pp.filter_genes(temp, min_cells=args.filter_min_cell)
        gene_count_after = len(list(temp.var_names) )
        print('Gene filtering done. Number of genes reduced from %d to %d'%(gene_count_before, gene_count_after))
        gene_ids = list(temp.var_names) 


        #print(len(gene_ids))

       	cell_barcode = np.array(temp.obs.index)
        temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(temp.X)))  #https://en.wikipedia.org/wiki/Quantile_normalization
        cell_vs_gene = np.transpose(temp)
        '''


        temp = sc.read_mtx(args.data_from)
        cell_vs_gene = sparse.csr_matrix.toarray(np.transpose(temp.X))
        cell_barcode = pd.read_csv('../data/CID44971_spatial/filtered_count_matrix/barcodes.tsv.gz', header=None)
        cell_barcode = np.array(list(cell_barcode[0]))
        gene_ids = pd.read_csv('../data/CID44971_spatial/filtered_count_matrix/features.tsv.gz', header=None)
        gene_ids = list(gene_ids[0])
        print("Number of genes %d"%len(gene_ids))
        print('Number of barcodes: %d'%cell_barcode.shape[0])
        print('Applying quantile normalization')
        temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  #https://en.wikipedia.org/wiki/Quantile_normalization
        cell_vs_gene = np.transpose(temp)  
        '''

        # now read the tissue position file. It has the format:     
        df = pd.read_csv(args.tissue_position_file, sep=",", header=None)   
        tissue_position = df.values
        barcode_vs_xy = dict() # record the x and y coordinates for each spot/cell
        for i in range (0, tissue_position.shape[0]):
            #barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][4], tissue_position[i][5]] # x and y coordinates
            barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][5], tissue_position[i][4]] #for some weird reason, in the .h5 format for LUAD sample, the x and y are swapped

       	coordinates = np.zeros((cell_barcode.shape[0], 2)) # insert the coordinates in the order of cell_barcodes
        for i in range (0, cell_barcode.shape[0]):
            coordinates[i,0] = barcode_vs_xy[cell_barcode[i]][0]
            coordinates[i,1] = barcode_vs_xy[cell_barcode[i]][1]

    
    
    ##################### make metadata: barcode_info ###################################
    i=0
    barcode_info=[]
    #cell_ROI = []
    for cell_code in cell_barcode:
        #print(coordinates[i,1])
        '''
        if coordinates[i,1]>3000 or coordinates[i,0]>3000:
            i=i+1
            continue
        cell_ROI.append(i)
        '''
        barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1], 0]) # last entry will hold the component number later
        i=i+1
    ################################################
    
    gene_info=dict()
    for gene in gene_ids:
        gene_info[gene]=''
    
    gene_index=dict()    
    i = 0
    for gene in gene_ids: 
        gene_index[gene] = i
        i = i+1

    ####################################################################
    # ligand - receptor database 
    print('ligand-receptor database reading.')
    df = pd.read_csv(args.database_path, sep=",")
    
    '''
            Ligand   Receptor          Annotation           Reference
    0        TGFB1     TGFBR1  Secreted Signaling      KEGG: hsa04350
    1        TGFB1     TGFBR2  Secreted Signaling      KEGG: hsa04350
    '''
    print('ligand-receptor database reading done.')
    print('Preprocess start.')
    ligand_dict_dataset = defaultdict(list)
    cell_cell_contact = dict() 
    count_pair = 0
    receptor_list = dict()
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        if ligand not in gene_info: # not found in the dataset
            continue    

        receptor = df["Receptor"][i]
        if receptor not in gene_info: # not found in the dataset
            continue   


        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = 'included'
        gene_info[receptor] = 'included'
        count_pair = count_pair + 1
        receptor_list[receptor] = ''
        if df["Annotation"][i] == 'Cell-Cell Contact':
            cell_cell_contact[receptor] = '' # keep track of which ccc are labeled as cell-cell-contact
    
    receptor_list = list(receptor_list.keys())
    ligand_list =  list(ligand_dict_dataset.keys())
    print('number of ligands %d '%len(ligand_dict_dataset.keys()))
    print('number of receptors %d '%len(receptor_list))
    included_gene=[]
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included':
            included_gene.append(gene)

    print('Total genes in this dataset: %d, number of genes working as ligand and/or receptor: %d '%(len(gene_ids),len(included_gene)))
    
    # assign id to each entry in the ligand-receptor database
    for ligand in ligand_dict_dataset:
        list_receptor = ligand_dict_dataset[ligand]
        list_receptor = np.unique(list_receptor)
        ligand_dict_dataset[ligand] = list_receptor
    

    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id 
            lr_id  = lr_id  + 1

    print("unique LR pair count %d"%lr_id)    

    #######################
    # load 'intra' database
    if True: #args.add_intra==1:
        receptor_intra = dict()
        for receptor in receptor_list:
            receptor_intra[ligand] = ''

        print('len receptor count %d'%len(receptor_intra))
        # keep only target species
        pathways_dict = defaultdict(list)
        count_kg = 0

        pathways = pd.read_csv(args.intra_human_ppi_path)        
        #pathways = pathways.drop_duplicates(ignore_index=True)
        for i in range (0, len(pathways)):
            source_gene = pathways['source'][i].upper()
            dest_gene = pathways['target'][i].upper()
            if source_gene in gene_info and dest_gene in gene_info:
                #if gene_info[source_gene] == 'included' and gene_info[dest_gene]=='included': #
                pathways_dict[source_gene].append([dest_gene, pathways['experimental_score'][i]])

        # Then make a kg for each ligand and save it
        for receptor_gene in receptor_intra:
            if receptor_gene in pathways_dict:
                print("####### %s found ###########"%receptor_gene)
                receptor_intra[receptor_gene] = pathways_dict[receptor_gene]
                count_kg = count_kg +1
            else:
                print("####### %s ###########"%receptor_gene)

        print('***** Total %d receptors have knowledge graph *****'%count_kg)

        pathways = pd.read_csv(args.intra_human_tf_target_path)        
        #pathways = pathways.drop_duplicates(ignore_index=True)
        for i in range (0, len(pathways)):
            source_gene = pathways['source'][i].upper()
            dest_gene = pathways['target'][i].upper()
            if source_gene in gene_info and dest_gene in gene_info:
                if pathways['mode'][i] == 1: #gene_info[source_gene] == 'included' and gene_info[dest_gene]=='included': # 
                    pathways_dict[source_gene].append([dest_gene, pathways['confidence_score'][i]])
                    

        # then make a kg for each receptor and save it

        for receptor_gene in receptor_intra:
            if receptor_gene in pathways_dict:
                print("####### %s found ###########"%receptor_gene)
                if (receptor_intra[receptor_gene]) != '':
                    receptor_intra[receptor_gene] = pathways_dict[receptor_gene]
                    count_kg = count_kg +1
            else:
                print("####### %s ###########"%receptor_gene)


       	print('***** Total %d ligands have knowledge graph *****'%count_kg) 
        with gzip.open(args.metadata_to +'/' + args.data_name + '_receptor_intra_KG.pkl', 'wb') as fp:  
            pickle.dump(receptor_intra, fp) 



