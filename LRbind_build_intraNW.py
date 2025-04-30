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

print('user input reading')
#current_dir = 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--data_name', type=str, help='Name of the dataset', required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--data_from', type=str, default='/cluster/projects/schwartzgroup/data/notta_pdac_visium_spaceranger_outputs_no_header/exp2_D1/outs/' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.',required=True) 
    #'../data/V1_Human_Lymph_Node_spatial/'
    ################# default is set ################################################################
    parser.add_argument( '--data_to', type=str, default='input_graph/', help='Path to save the input graph (to be passed to GAT)')
    parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
    parser.add_argument( '--filter_min_cell', type=int, default=1 , help='Minimum number of cells for gene filtering') 
    parser.add_argument( '--threshold_gene_exp', type=float, default=98, help='Threshold percentile for gene expression. Genes above this percentile are considered active.')
    parser.add_argument( '--tissue_position_file', type=str, default='None', help='If your --data_from argument points to a *.mtx file instead of Space Ranger, then please provide the path to tissue position file.')
    parser.add_argument( '--spot_diameter', type=float, default=89.43, help='Spot/cell diameter for filtering ligand-receptor pairs based on cell-cell contact information. Should be provided in the same unit as spatia data (for Visium, that is pixel).')
    parser.add_argument( '--split', type=int, default=0 , help='How many split sections?') 
    parser.add_argument( '--neighborhood_threshold', type=float, default=0 , help='Set neighborhood threshold distance in terms of same unit as spot diameter') 
    parser.add_argument( '--num_hops', type=int, default=4 , help='Number of hops for direct connection')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.') 
    parser.add_argument( '--remove_LR', type=str, help='Test LR to predict')
    #parser.add_argument( '--remove_LR', type=str, default=[['CCL19', 'CCR7']], help='Test LR to predict') #, required=True) # FN1-RPSA
    parser.add_argument( '--target_lig', type=str, default="", help='Test LR to predict')
    parser.add_argument( '--target_rec', type=str, default="", help='Test LR to predict')
    parser.add_argument( '--remove_lig', type=str, default="False", help='Test LR to predict')
    parser.add_argument( '--remove_rec', type=str, default="False", help='Test LR to predict')
    parser.add_argument( '--remove_lrp', type=str, default="True", help='remove target LR pair from database')
    parser.add_argument( '--add_intra', type=int, default=-1, help='Set to 1 if you want to add intra network')
    parser.add_argument( '--intra_cutoff', type=float, default=0.3 , help='?') 
    args = parser.parse_args()
    
    args.remove_LR = [[args.target_lig, args.target_rec]]

    if args.remove_rec == "True" and args.target_rec == "":
        print("Please input args.target_rec, or set args.remove_rec=False")
        exit()

    if args.remove_lig == "True" and args.target_lig == "":
        print("Please input args.target_lig, or set args.remove_lig=False")
        exit()

    if args.neighborhood_threshold == 0:
        args.neighborhood_threshold = args.spot_diameter*args.num_hops

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
        
    
    
    #######################
    if args.target_lig in gene_ids:
        print('target ligand exist in the gene list')
    if args.target_rec in gene_ids:
        print('target rec exist in the gene list')


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
        
    ####################### target LR list############################################
    target_LR_index = dict() 
    discard_genes = dict()
    target_LR_list = args.remove_LR #[['CCL19', 'CCR7']]
    for target_LR in target_LR_list:
        ligand = target_LR[0]
        receptor = target_LR[1]
        target_LR_index[ligand + '+' + receptor] = [gene_index[ligand], gene_index[receptor]]
        discard_genes[ligand]= ''
        discard_genes[receptor]= ''

    print(target_LR_index.keys())
       
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


        if args.remove_lrp == "True":
            if ligand+'+'+receptor in target_LR_index:
                continue
        if args.remove_lig == "True" and ligand in args.target_lig:
            print('remove_lig true')
            continue

        if args.remove_rec == "True" and receptor in args.target_rec:
            print('remove_rec true')
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
    print('number of ligand-receptor pairs in this dataset %d '%count_pair) 
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
    cell_gene_set = gene_ids # ligand_list + receptor_list
    df = defaultdict(list)
    for gene in cell_gene_set:
        j = gene_index[gene] 
        df[gene_ids[j]]=list(cell_vs_gene[:, j])

    data = pd.DataFrame(df)
    '''if os.path.isfile(args.metadata_to +'/' + args.data_name + 'gene_coexpression_matrix.pkl'):
        print('Reading gene_coexpression_matrix calculation')
        with gzip.open(args.metadata_to +'/' + args.data_name + 'gene_coexpression_matrix.pkl', 'rb') as fp:  
    	    gene_coexpression_matrix = pickle.load(fp)
    else: ''' 
    
    print('Running gene_coexpression_matrix calculation')
    gene_coexpression_matrix = data.corr(method='pearson')
    with gzip.open(args.metadata_to +'/' + args.data_name + 'gene_coexpression_matrix.pkl', 'wb') as fp:  
        pickle.dump(gene_coexpression_matrix, fp)


  
