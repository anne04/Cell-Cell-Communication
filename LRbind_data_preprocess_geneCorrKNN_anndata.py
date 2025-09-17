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
from sklearn.metrics.pairwise import euclidean_distances
#import pathway_search_LRbind as pathway
from sklearn.neighbors import NearestNeighbors

print('user input reading')
#current_dir = 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--data_name', type=str, default='LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir', help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    parser.add_argument( '--data_from', type=str, default='../data/LUAD/LUAD_GSM5702473_TD1/' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    #'../data/V1_Human_Lymph_Node_spatial/'
    ################# default is set ################################################################
    parser.add_argument( '--data_to', type=str, default='input_graph/', help='Path to save the input graph (to be passed to GAT)')
    parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
    parser.add_argument( '--filter_min_cell', type=int, default=1 , help='Minimum number of cells for gene filtering') 
    parser.add_argument( '--threshold_gene_exp', type=float, default=97, help='Threshold percentile for gene expression. Genes above this percentile are considered active.')
    parser.add_argument( '--tissue_position_file', type=str, default='None', help='If your --data_from argument points to a *.mtx file instead of Space Ranger, then please provide the path to tissue position file.')
    parser.add_argument( '--spot_diameter', type=float, default=160, help='Spot/cell diameter for filtering ligand-receptor pairs based on cell-cell contact information. Should be provided in the same unit as spatia data (for Visium, that is pixel).')
    parser.add_argument( '--split', type=int, default=0 , help='How many split sections?') 
    parser.add_argument( '--neighborhood_threshold', type=float, default=0 , help='Set neighborhood threshold distance in terms of same unit as spot diameter') 
    parser.add_argument( '--distance_measure', type=str, default='fixed' , help='Set neighborhood cutoff criteria. Choose from [knn, fixed]')
    parser.add_argument( '--k', type=int, default=30 , help='Set neighborhood cutoff number. This will be used if --distance_measure=knn')    
    parser.add_argument( '--juxtacrine_distance', type=float, default=-1, help='Distance for filtering ligand-receptor pairs based on cell-cell contact information. Automatically calculated unless provided. It has the same unit as the coordinates (for Visium, that is pixel).')
    parser.add_argument( '--num_hops', type=int, default=3 , help='Number of hops for direct connection')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database_no_predictedPPI.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.') 
    parser.add_argument( '--remove_LR', type=str, help='Test LR to predict')
    #parser.add_argument( '--remove_LR', type=str, default=[['CCL19', 'CCR7']], help='Test LR to predict') #, required=True) # FN1-RPSA
    parser.add_argument( '--target_lig', type=str, default="TGFB1", help='Test LR to predict')
    parser.add_argument( '--target_rec', type=str, default="ACVRL1", help='Test LR to predict')
    parser.add_argument( '--remove_lig', type=str, default="False", help='Test LR to predict')
    parser.add_argument( '--remove_rec', type=str, default="False", help='Test LR to predict')
    parser.add_argument( '--remove_lrp', type=str, default="True", help='remove target LR pair from database')
    parser.add_argument( '--add_intra', type=int, default=1, help='Set to 1 if you want to add intra network')
    parser.add_argument( '--intra_cutoff', type=float, default=0.3 , help='?') 
    parser.add_argument( '--threshold_gene_exp_intra', type=float, default=70, help='Threshold percentile for gene expression. Genes above this percentile are considered active.')
    parser.add_argument( '--local_coexpression', type=int, default=0, help='Set to 1 if you want to use local coexpression matrix.')
    parser.add_argument( '--add_negatome', type=int, default=0, help='Set to 1 if you want to add negatome')
    parser.add_argument( '--check_protien_emb', type=int, default=0, help='Set to 1 if you want to see how many of your genes have protein emb')
    parser.add_argument( '--block_autocrine', type=int, default=0 , help='Set to 1 if you want to ignore autocrine signals.') 
    parser.add_argument( '--block_juxtacrine', type=int, default=0, help='Set to 1 if you want to ignore juxtacrine signals.')  
    args = parser.parse_args()
    
    args.remove_LR = [[args.target_lig, args.target_rec]]

    if args.remove_rec == "True" and args.target_rec == "":
        print("Please input args.target_rec, or set args.remove_rec=False")
        exit()

    if args.remove_lig == "True" and args.target_lig == "":
        print("Please input args.target_lig, or set args.remove_lig=False")
        exit()

    #if args.neighborhood_threshold == 0:
    #    args.neighborhood_threshold = args.spot_diameter*args.num_hops

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
    adata_h5 = sc.read_h5ad(args.data_from)
    print('input data read done')
    gene_count_before = len(list(adata_h5.var_names))    
    sc.pp.filter_genes(adata_h5, min_cells=args.filter_min_cell)
    gene_count_after = len(list(adata_h5.var_names) )  
    print('Gene filtering done. Number of genes reduced from %d to %d'%(gene_count_before, gene_count_after))
    gene_ids = list(adata_h5.var_names)
    coordinates = adata_h5.obsm['spatial']
    cell_barcode = np.array(adata_h5.obs_names) #obs.index)
    print('Number of barcodes: %d'%cell_barcode.shape[0])
    print('Applying quantile normalization')
    temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  #https://en.wikipedia.org/wiki/Quantile_normalization
    cell_vs_gene = np.transpose(temp)


    with gzip.open(args.data_to + args.data_name + '_cell_vs_gene_quantile_transformed', 'wb') as fp:
        pickle.dump(cell_vs_gene, fp)

    print('saved: '+args.data_to + args.data_name + '_cell_vs_gene_quantile_transformed')    
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

    with gzip.open(args.data_to + args.data_name + '_gene_index', 'wb') as fp:
        pickle.dump([gene_index, gene_ids, cell_barcode], fp)

    print('gene_index saved')        
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
#    print('number of ligand-receptor pairs in this dataset %d '%count_pair) 
    print('number of ligands %d '%len(ligand_dict_dataset.keys()))
    print('number of receptors %d '%len(receptor_list))
    included_gene=[]
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included':
            included_gene.append(gene)
            
    print('Total genes in this dataset: %d, number of genes working as ligand and/or receptor: %d '%(len(gene_ids),len(included_gene)))


    ####################### for debug purpose ###########
    if args.check_protien_emb == 1:
        print('Looking for embedding of ligand and receptor genes')
        with gzip.open('database/ligand_receptor_protein_embedding.pkl', 'rb') as fp:  
            gene_vs_embedding = pickle.load(fp)    

        for gene in ligand_list:
            if gene not in gene_vs_embedding:
                print('Embedding not found for ligand:' + gene)

        for gene in receptor_list:
            if gene not in gene_vs_embedding:
                print('Embedding not found for receptor:' + gene)




    #exit(0)
    #######################



    
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
    ##################### read negatome to see how many overlaps ####################
    if args.add_negatome == 1:
        with gzip.open('database/negatome_gene_complex_set', 'rb') as fp:  
            negatome_gene, negatome_lr_unique = pickle.load(fp)

        
        set_1 = set(gene_ids)
        set_2 = set(negatome_gene)
        
        intersection_list = list(set_1.intersection(set_2))
        print('len of common genes with negatome genes %d'%len(intersection_list))
        intersection_list = intersection_list #[0:len(intersection_list)//2]
        neg_l_r_pair = defaultdict(dict)
        neg_l_r_pair_count = 0
        neg_chain_A = []
        neg_chain_B = []
        for neg_lr_pair in negatome_lr_unique: 
            #if neg_l_r_pair_count==(len(negatome_lr_unique)//2):
            #    break
            #print(neg_lr_pair)
            neg_lig = neg_lr_pair.split('_with_')[0]
            neg_rec = neg_lr_pair.split('_with_')[1]
            if neg_lig in intersection_list and neg_rec in intersection_list:
                neg_l_r_pair[neg_lig][neg_rec] = 1
                neg_l_r_pair_count = neg_l_r_pair_count + 1
                neg_chain_A.append(neg_lig)
                neg_chain_B.append(neg_rec)

        print('unique negatome lr pair count is %d'%neg_l_r_pair_count)
        neg_chain_A = list(np.unique(neg_chain_A))
        neg_chain_B = list(np.unique(neg_chain_B))

        with gzip.open('database/negatome_ligand_receptor_set', 'rb') as fp:  
            negatome_ligand_list, negatome_receptor_list, lr_unique_negatome = pickle.load(fp)
            
        found_overlap = 0  
        for ligand in lr_unique_negatome:
            if ligand in l_r_pair:
                for receptor in lr_unique_negatome[ligand]:
                    #print(receptor)
                    if receptor in l_r_pair[ligand]:
                        found_overlap = found_overlap + 1
                        #print(ligand + '_to_' + receptor)
        
        print('found overlap %d'%found_overlap)
                
        count = 0
        for ligand in negatome_ligand_list:
            if ligand in ligand_list:
                count = count + 1

        print('negatome ligand found %d'%count)

        count = 0
        for receptor in negatome_receptor_list:
            if receptor in receptor_list:
                count = count + 1

        print('negatome receptor found %d'%count)


    #exit(0)    



    ##################################################################################
    #coordinates = coordinates[cell_ROI]
    #print(cell_ROI)
    #print(coordinates.shape)
    #cell_vs_gene = cell_vs_gene[cell_ROI,:]
    #print(cell_vs_gene.shape)


    ###################################################################################
    # build physical distance matrix
    print('Build physical distance matrix')
    if args.distance_measure == 'fixed':
        # then you need to set the args.neighborhood_threshold
        if args.neighborhood_threshold == 0:
            from sklearn.metrics.pairwise import euclidean_distances
            distance_matrix = euclidean_distances(coordinates, coordinates)
            #### then automatically calculate the min distance between two nodes #########
            sorted_first_row = np.sort(distance_matrix[0,:])
            distance_a_b = sorted_first_row[1]
            args.neighborhood_threshold = distance_a_b * 4 # 4 times the distance between two nodes
            distance_matrix = 0
            gc.collect()

        nbrs = NearestNeighbors(radius=args.neighborhood_threshold, algorithm='kd_tree', n_jobs=-1)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        print('Neighborhood distance is set to be %g (same unit as the coordinates)'%(args.neighborhood_threshold))
    else:
        print('Neighborhood distance is set to be %d nearest neighbors'%args.k)
        nbrs = NearestNeighbors(n_neighbors=args.k, algorithm='kd_tree', n_jobs=-1)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

    unique_distances = np.unique(distances)
    distance_a_b = unique_distances[1]
    # distances: array of shape (n_cells, k) with the Euclidean distance from 
    # each cell to its k neighbors.
    # indices: array of shape (n_cells, k) with the neighbor indices (row indices of X).
    # each cell 'j' will receive signal from its neighbors 'i' in descending order of weights
    print('Assign weight to the neighborhood relations based on neighborhood distance')
    weightdict_i_to_j = defaultdict(dict)
    for cell_idx in range (0, indices.shape[0]):
        max_value = np.max(distances[cell_idx,:])
        min_value = np.min(distances[cell_idx,:])
        for neigh_idx in range (0, indices.shape[1]):
            neigh_cell_idx = indices[cell_idx][neigh_idx]
            distance_neigh_cell = distances[cell_idx][neigh_idx]
            flipped_distance_neigh_cell = 1-(distance_neigh_cell-min_value)/(max_value-min_value)
            # i = neigh_cell_idx, j = cell_idx
            weightdict_i_to_j[neigh_cell_idx][cell_idx] = flipped_distance_neigh_cell



    #### automatically set the args.juxtacrine_distance ############
    if args.juxtacrine_distance == -1: 
        args.juxtacrine_distance = distance_a_b

    if args.block_juxtacrine == 0:
        print("Auto calculated juxtacrine distance is %g. To change it use --juxtacrine_distance"%args.juxtacrine_distance)




    
    #####################################################################################
    # Set threshold gene percentile
    cell_percentile = []
    for i in range (0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i]) # sort each row/cell in ascending order of gene expressions
        ## inter ##
        active_cutoff = np.percentile(y, args.threshold_gene_exp)
        if active_cutoff == min(cell_vs_gene[i][:]):
            times = 1
            while active_cutoff == min(cell_vs_gene[i][:]):
                new_threshold = args.threshold_gene_exp + 5 * times                    
                if new_threshold >= 100:
                    active_cutoff = max(cell_vs_gene[i][:])  
                    break
                active_cutoff = np.percentile(y, new_threshold)
                times = times + 1 

        cell_percentile.append(active_cutoff)     


    ############################################################################################
    # for each cell, record the active genes
    
    if args.add_intra == 1:
        intra_active = []
        for i in range (0, cell_vs_gene.shape[0]):
            y = sorted(cell_vs_gene[i])
             ## intra ##
            active_cutoff = np.percentile(y, args.threshold_gene_exp_intra) 
            if active_cutoff == min(cell_vs_gene[i][:]):
                times = 1
                while active_cutoff == min(cell_vs_gene[i][:]):
                    new_threshold = args.threshold_gene_exp_intra + 5 * times
                    if new_threshold >= 100:
                        active_cutoff = max(cell_vs_gene[i][:])  
                        break
                    active_cutoff = np.percentile(y, new_threshold)
                    times = times + 1 
            
            intra_active.append(active_cutoff)


    ##################### target LR cell pairs #########################################################
    target_cell_pair = defaultdict(list)
    debug = dict()
    for target_LR in target_LR_list:
        ligand = target_LR[0]
        receptor = target_LR[1]
        for i in range (0, cell_vs_gene.shape[0]): # ligand
            if cell_vs_gene[i][gene_index[ligand]] < cell_percentile[i]:
                continue
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                if cell_vs_gene[j][gene_index[receptor]] < cell_percentile[j]:
                    continue
                if i not in weightdict_i_to_j or j not in weightdict_i_to_j[i]: #dist_X[i,j]==0:
                    continue
                    
                target_cell_pair[ligand+'+'+receptor].append([i, j])    
                debug[i] = ''
                debug[j] = ''
    print('target_cell_pair %d'%len(debug.keys()))  
    ##############################################################################
    # some preprocessing before making the input graph
    blocked_gene_per_cell = defaultdict(dict)

    ###############################################################################
    count_total_edges = 0
    cells_ligand_vs_receptor = defaultdict(dict)
    """
    cells_ligand_vs_receptor = []
    for i in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor.append([])
        
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[0]):
            cells_ligand_vs_receptor[i].append([])
            cells_ligand_vs_receptor[i][j] = []
    """
    #ligand_list =  list(ligand_dict_dataset.keys())            
    start_index = 0 #args.slice
    end_index = len(ligand_list) #min(len(ligand_list), start_index+100)
    #inactive_node=[]
    for g in range(start_index, end_index): 
        gene = ligand_list[g]

        #for i in range (0, cell_vs_gene.shape[0]): # ligand
        for i in weightdict_i_to_j.keys():
            if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i]:
                continue

            if gene_index[gene] in blocked_gene_per_cell[i]:
                continue

            for j in weightdict_i_to_j[i].keys():
            #for j in range (0, cell_vs_gene.shape[0]): # receptor
                #if i not in weightdict_i_to_j or j not in weightdict_i_to_j[i]: #dist_X[i,j]==0: 
                #    continue    
                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j]: # or cell_vs_gene[i][gene_index[gene]] >= cell_percentile[i][4] :#gene_list_percentile[gene_rec][1]: #global_percentile: #
            
                        if gene_index[gene_rec] in blocked_gene_per_cell[j]:
                            continue
                        
                        if (gene_rec in cell_cell_contact) and (args.block_juxtacrine==1 or euclidean_distances(coordinates[i],coordinates[j]) > args.juxtacrine_distance):
                            continue
    
                        communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                        relation_id = l_r_pair[gene][gene_rec]
    
                        if communication_score<=0:
                            print('zero valued ccc score found. Might be a potential ERROR!! ')
                            continue	
                            
                        if i in cells_ligand_vs_receptor:
                            if j in cells_ligand_vs_receptor[i]:
                                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])

                            else:
                                cells_ligand_vs_receptor[i][j] = []
                                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])

                        else:
                            cells_ligand_vs_receptor[i][j] = []        
                            cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])

                        count_total_edges = count_total_edges + 1
    
        print('%d/%d ligand genes processed'%(g+1, len(ligand_list)), end='\r')
    
    print('') 
    #print('total number of edges in the input graph %d '%count_total_edges)
    ################################################################################
    # input graph generation
    ccc_index_dict = dict()
    row_col = [] # list of input edges, row = from node, col = to node
    edge_weight = [] # 3D edge features in the same order as row_col
    lig_rec = [] # ligand and receptors corresponding to the edges in the same order as row_col
    self_loop_found = defaultdict(dict) # to keep track of self-loops -- used later during visualization plotting
    for i in cells_ligand_vs_receptor.keys():
        #ccc_j = []
        for j in cells_ligand_vs_receptor[i].keys():
            if i in weightdict_i_to_j and j in weightdict_i_to_j[i]: #dist_X[i,j]>0: #distance_matrix[i][j] <= args.neighborhood_threshold: 
                count_local = 0
                if len(cells_ligand_vs_receptor[i][j])>0:
                    for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                        gene = cells_ligand_vs_receptor[i][j][k][0]
                        gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                        ligand_receptor_coexpression_score = cells_ligand_vs_receptor[i][j][k][2]

                        row_col.append([i,j])
                        edge_weight.append([weightdict_i_to_j[i][j]]) #, 2]) #, cells_ligand_vs_receptor[i][j][k][3]])
                        lig_rec.append([gene, gene_rec])

                        row_col.append([j,i])
                        edge_weight.append([weightdict_i_to_j[j][i]]) #, 2]) #, cells_ligand_vs_receptor[i][j][k][3]])
                        #edge_weight.append([dist_X[i,j], ligand_receptor_coexpression_score, cells_ligand_vs_receptor[i][j][k][3]])
                        lig_rec.append([gene_rec, gene])
                                                  
                        if i==j: # self-loop
                            self_loop_found[i][j] = ''
    

    total_num_cell = cell_vs_gene.shape[0]
    print('total number of spots/cells is %d, and edges is %d in the input graph'%(total_num_cell, len(row_col)))
    #print('preprocess done.')
    #print('writing data ...')

    ################## input gene graph #################################################
    lig_rec_dict = defaultdict(dict)    
    for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        if i in lig_rec_dict:
            if j in lig_rec_dict[i]:
                lig_rec_dict[i][j].append(lig_rec[index]) 
            else:
                lig_rec_dict[i][j] = []
                lig_rec_dict[i][j].append(lig_rec[index])
        else:
            lig_rec_dict[i][j] = []
            lig_rec_dict[i][j].append(lig_rec[index])                

    gene_type_id = 0
    gene_type = dict()

    if (args.remove_lig == "True" or args.target_lig not in ligand_list) and args.target_lig != "":
        ligand_list.append(args.target_lig)
        print('adding ligand %s to ligand list'%args.target_lig)

    if (args.remove_rec == "True" or args.target_rec not in receptor_list) and args.target_rec != "":
        receptor_list.append(args.target_rec)
        print('adding receptor %s to receptor list'%args.target_rec)
        
    for gene in ligand_list:
        gene_type[gene] = gene_type_id
        gene_type_id = gene_type_id + 1
        
    for gene in receptor_list:
        gene_type[gene] = gene_type_id
        gene_type_id = gene_type_id + 1
          
        
    gene_node_index = 0
    gene_node_list_per_spot = defaultdict(dict)
    gene_node_type = []
    barcode_info_gene = []
    gene_node_expression = []
    for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        ligand_name = lig_rec[index][0]
        receptor_name = lig_rec[index][1]
        
        # add ligand gene node to spot/cell i
        spot_id = i
        gene = ligand_name
        if spot_id not in gene_node_list_per_spot or gene not in gene_node_list_per_spot[spot_id]: 
            gene_node_list_per_spot[spot_id][gene] = gene_node_index
            gene_node_type.append(gene_type[gene])        
            barcode_info_gene.append([barcode_info[spot_id][0], barcode_info[spot_id][1], barcode_info[spot_id][2], barcode_info[spot_id][3], gene_node_index, gene])
            gene_node_expression.append(cell_vs_gene[spot_id][gene_index[gene]])
            gene_node_index = gene_node_index + 1
    
        # add receptor gene node to spot/cell j
        spot_id = j
        gene = receptor_name
        if spot_id not in gene_node_list_per_spot or gene not in gene_node_list_per_spot[spot_id]:
            gene_node_list_per_spot[spot_id][gene] = gene_node_index
            gene_node_type.append(gene_type[gene])        
            barcode_info_gene.append([barcode_info[spot_id][0], barcode_info[spot_id][1], barcode_info[spot_id][2], barcode_info[spot_id][3], gene_node_index, gene])
            gene_node_expression.append(cell_vs_gene[spot_id][gene_index[gene]])
            gene_node_index = gene_node_index + 1
        
    
 
     
    print('Total number of unique gene node types is %d'%np.max(np.unique(gene_node_type)))
    #print(np.unique(gene_node_type))
    # old edges replacement with gene nodes
    row_col_gene = []
    #edge_weight_gene = []
    gene_node_index_active = dict()
    for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        # i to j. j's emb is updated through this edge, not i's. So, j is active
        ligand_gene = lig_rec[index][0]
        receptor_gene = lig_rec[index][1]
        gene_node_from = gene_node_list_per_spot[i][ligand_gene]
        gene_node_to = gene_node_list_per_spot[j][receptor_gene]
        row_col_gene.append([gene_node_from, gene_node_to])
        #gene_node_index_active[gene_node_from] = ''
        gene_node_index_active[gene_node_to] = ''
        # edge_weight_gene.append(edge_weight[index])

    print('Total number of gene nodes in this graph is %d, inactive %d, active %d'%(gene_node_index, gene_node_index-len(gene_node_index_active.keys()),len(gene_node_index_active.keys())))


    start_of_intra_edge = len(edge_weight) 
    # add the ligands and receptors from negatome

    ### negatome ###
    if args.add_negatome == 1:
        with gzip.open('database/negatome_ligand_receptor_set', 'rb') as fp:  
            negatome_ligand_list, negatome_receptor_list, lr_unique = pickle.load(fp)
        
        for ligand_gene in negatome_ligand_list:
            if ligand_gene in gene_ids and ligand_gene not in ligand_list:    
                ligand_list.append(ligand_gene)

        for receptor_gene in negatome_receptor_list:
            if receptor_gene in gene_ids and receptor_gene not in receptor_list:    
                receptor_list.append(receptor_gene)


        for ligand_gene in neg_chain_A:
            if ligand_gene in gene_ids and ligand_gene not in ligand_list:    
                ligand_list.append(ligand_gene)

        for receptor_gene in neg_chain_B:
            if receptor_gene in gene_ids and receptor_gene not in receptor_list:    
                receptor_list.append(receptor_gene)


        

    ################


    cell_gene_set = ligand_list + receptor_list #list(active_genes.keys()) #ligand_list + receptor_list



    print('gene count for corr matrix %d'%len(cell_gene_set))
    if args.local_coexpression == 0:
        df = defaultdict(list)
        for gene in cell_gene_set:
            j = gene_index[gene] 
            df[gene_ids[j]]=list(cell_vs_gene[:, j])
    
        data = pd.DataFrame(df)
        print('Running gene_coexpression_matrix calculation')
        gene_coexpression_matrix = data.corr(method='spearman')
       


    start_of_intra_edge = len(edge_weight)
    print("start_of_intra_edge %d"%(start_of_intra_edge))
    for i in weightdict_i_to_j.keys():
        if args.local_coexpression == 1:
            neighbor_list = list(weightdict_i_to_j[i].keys()) #neighbor_list_per_cell[i]
            temp_cell_vs_gene = cell_vs_gene[neighbor_list]
            df = defaultdict(list)
            for gene in cell_gene_set:
                j = gene_index[gene] 
                df[gene_ids[j]]=list(temp_cell_vs_gene[:, j])
        
            data = pd.DataFrame(df)
            print('Running gene_coexpression_matrix calculation')
            gene_coexpression_matrix = data.corr(method='spearman')
        
        spot_id = i
        print('i %d, edge %d, gene node %d'%(i, len(row_col_gene), len(gene_node_type)))
        cell_intra_gcm = defaultdict(dict)
        for gene_a in cell_gene_set:
            if gene_index[gene_a] in blocked_gene_per_cell[i]:
                continue
            #if gene_a == "CCL19" :
            #    print("found ccl19")
            if cell_vs_gene[spot_id][gene_index[gene_a]] < intra_active[spot_id]: #cell_percentile[spot_id]:
                continue
            gene_to_connect = []
            

            #sorted_gene_b = []
            #if gene_a in ligand_list:
                # find knn ligand genes
            sorted_gene_b_ligand = []
            for gene_b in ligand_list:
                if gene_b==gene_a or gene_coexpression_matrix[gene_a][gene_b]<=0: # or cell_vs_gene[spot_id][gene_index[gene_b]] < intra_active[spot_id]: #cell_percentile[spot_id]:
                    continue    
                sorted_gene_b_ligand.append([gene_b, gene_coexpression_matrix[gene_a][gene_b]])
            
            sorted_gene_b_ligand = sorted(sorted_gene_b_ligand, key = lambda x: x[1], reverse=True) # highest to lowest coeff
            sorted_gene_b_ligand = sorted_gene_b_ligand[0:5]
                   
            #if gene_a in receptor_list: 
                # find knn ligand genes 
            sorted_gene_b_receptor = []
            for gene_b in receptor_list:
                    
                if gene_b==gene_a or gene_coexpression_matrix[gene_a][gene_b]<=0: # or cell_vs_gene[spot_id][gene_index[gene_b]] < intra_active[spot_id]: #cell_percentile[spot_id]:
                    continue 

                sorted_gene_b_receptor.append([gene_b, gene_coexpression_matrix[gene_a][gene_b]])

            sorted_gene_b_receptor = sorted(sorted_gene_b_receptor, key = lambda x: x[1], reverse=True) # highest to lowest coeff
            sorted_gene_b_receptor = sorted_gene_b_receptor[0:5]

            #print('len gene_b list %d'%len(sorted_gene_b))
            sorted_gene_b = sorted_gene_b_ligand + sorted_gene_b_receptor
            sorted_gene_b_temp = []
            for item in sorted_gene_b:
                sorted_gene_b_temp.append(item[0])

            sorted_gene_b = sorted_gene_b_temp
            
            for gene_b in sorted_gene_b: #cell_gene_set:
                if gene_index[gene_b] in blocked_gene_per_cell[i]:
                    continue

                if gene_a in cell_intra_gcm and gene_b in cell_intra_gcm[gene_a]:
                    continue


                #if cell_vs_gene[spot_id][gene_index[gene_b]] < cell_percentile[spot_id]:
                #    continue

                if gene_coexpression_matrix[gene_a][gene_b] < args.intra_cutoff: #0.30:
                    continue
                
                if spot_id not in gene_node_list_per_spot or gene_a not in gene_node_list_per_spot[spot_id]:
                    gene_node_list_per_spot[spot_id][gene_a] = gene_node_index 
                    barcode_info_gene.append([barcode_info[spot_id][0], barcode_info[spot_id][1], barcode_info[spot_id][2], barcode_info[spot_id][3], gene_node_index, gene_a])
                    gene_node_expression.append(cell_vs_gene[spot_id][gene_index[gene_a]])
                    gene_node_index = gene_node_index + 1
                    # if gene_a is of new type, add it to the dictionary
                    if gene_a not in gene_type:
                        gene_type[gene_a] = gene_type_id 
                        gene_type_id = gene_type_id + 1
                        
                    gene_node_type.append(gene_type[gene_a])        
                    

                gene_a_idx = gene_node_list_per_spot[spot_id][gene_a]   
                
                if spot_id not in gene_node_list_per_spot or gene_b not in gene_node_list_per_spot[spot_id]:
                    gene_node_list_per_spot[spot_id][gene_b] = gene_node_index 
                    barcode_info_gene.append([barcode_info[spot_id][0], barcode_info[spot_id][1], barcode_info[spot_id][2], barcode_info[spot_id][3], gene_node_index, gene_b])
                    gene_node_expression.append(cell_vs_gene[spot_id][gene_index[gene_b]])
                    gene_node_index = gene_node_index + 1
                    # if gene_b is of new type, add it to the dictionary
                    if gene_b not in gene_type:
                        gene_type[gene_b] = gene_type_id 
                        gene_type_id = gene_type_id + 1
                        
                    gene_node_type.append(gene_type[gene_b])        

                
                gene_b_idx = gene_node_list_per_spot[spot_id][gene_b]
                
                
                
                if True: #gene_a_idx not in gene_node_index_active:
                    row_col_gene.append([gene_b_idx, gene_a_idx])
                    edge_weight.append([gene_coexpression_matrix[gene_b][gene_a]])
                    lig_rec.append([gene_b, gene_a])  
                    gene_node_index_active[gene_a_idx] = ''
                    cell_intra_gcm [gene_b][gene_a] = 1

                if True: #gene_b_idx not in gene_node_index_active:
                    row_col_gene.append([gene_a_idx, gene_b_idx])
                    edge_weight.append([gene_coexpression_matrix[gene_a][gene_b]])
                    lig_rec.append([gene_a, gene_b])
                    gene_node_index_active[gene_b_idx] = ''
                    cell_intra_gcm [gene_a][gene_b] = 1
             
                #print('total edges: %d, total gene nodes %d'%(len(row_col_gene), gene_node_index))
   
                
    
    gene_node_index_active = dict()
    rec_active_count = defaultdict(list)
    ligand_active_count = defaultdict(list)
    for index in range (0, len(row_col_gene)):
        #i = row_col_gene[index][0]
        j = row_col_gene[index][1]
        #gene_node_index_active[i] = ''
        gene_node_index_active[j] = ''
        if lig_rec[index][1] == args.target_lig:
            ligand_active_count[j].append(1)
        elif lig_rec[index][1] == args.target_rec:
            rec_active_count[j].append(1)

    total_incoming_ligand = 0
    for j in ligand_active_count:
        ligand_active_count[j] = np.sum(ligand_active_count[j])
        total_incoming_ligand = total_incoming_ligand + ligand_active_count[j]

    total_incoming_rec = 0
    for j in rec_active_count:
        rec_active_count[j] = np.sum(rec_active_count[j])
        total_incoming_rec = total_incoming_rec + rec_active_count[j]

    total_num_gene_node = len(gene_node_type)
    print('Total number of gene nodes in this graph is %d, inactive %d, active %d'%(gene_node_index, gene_node_index-len(gene_node_index_active.keys()),len(gene_node_index_active.keys())))
    print('active '+args.target_lig +' node %d, with number of incoming connections %d'%(len(ligand_active_count), total_incoming_ligand))
    print('active '+args.target_rec +' node %d, with number of incoming connections %d'%(len(rec_active_count), total_incoming_rec))
    print('inter edge count %d, intra edge count %d\n'%(start_of_intra_edge, len(row_col_gene)-start_of_intra_edge))    

            
    with gzip.open(args.data_to + args.data_name + '_adjacency_gene_records', 'wb') as fp:  
        pickle.dump([row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge], fp)

    #### needed if split data is used ##############
    if args.split>0:
        i=0
        node_id_sorted_xy=[]
        for gene_node in barcode_info_gene:
            node_id_sorted_xy.append([gene_node[4], gene_node[1],gene_node[2]])
            i=i+1
        	
        node_id_sorted_xy = sorted(node_id_sorted_xy, key = lambda x: (x[1], x[2]))
        with gzip.open(args.metadata_to + args.data_name+'_'+'gene_node_id_sorted_xy', 'wb') as fp:  #b, a:[0:5]   
        	pickle.dump(node_id_sorted_xy, fp)
    


    with gzip.open(args.metadata_to + args.data_name +'_barcode_info', 'wb') as fp:  
        pickle.dump(barcode_info, fp)

    with gzip.open(args.metadata_to + args.data_name +'_barcode_info_gene', 'wb') as fp:  
        pickle.dump([barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, weightdict_i_to_j, l_r_pair, gene_node_index_active, ligand_active_count, rec_active_count], fp) #, ligand_active_count, rec_active_count

    with gzip.open(args.metadata_to + args.data_name +'_test_set', 'wb') as fp:  
        pickle.dump([target_LR_index, target_cell_pair], fp)

    
    
    ######### optional #############################################################           
    # we do not need this to use anywhere. But just for debug purpose we are saving this. We can skip this if we have space issue.
    
    #with gzip.open(args.data_to + args.data_name + '_cell_vs_gene_quantile_transformed', 'wb') as fp:  
    #	pickle.dump(cell_vs_gene, fp)
     
    print('write data done')

    if args.add_negatome == 1:
        with gzip.open('database/negatome_gene_complex_set', 'rb') as fp:  
            negatome_gene, negatome_lr_unique = pickle.load(fp)
          
        count = 0
        negatome_unique_pair = dict()
        for i in range (0, len(barcode_info)):
            ligand_node_index = []
            for gene in gene_node_list_per_spot[i]:
                if gene in ligand_list:
                    ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])
            
            receptor_node_index_intra = []
           
            for gene in gene_node_list_per_spot[i]:
                if gene in receptor_list:
                    receptor_node_index_intra.append([gene_node_list_per_spot[i][gene], gene])

            
            for i_gene in ligand_node_index:  
                for j_gene in receptor_node_index_intra:
                    if i_gene[1]==j_gene[1]:
                        continue
                  
                    if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique:
                        #temp = distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) # 
                        #dot_prod_list_negatome_intra.append([temp, i, i, i_gene[1], j_gene[1], i_gene[0], j_gene[0]])
                        count = count+1
                        negatome_unique_pair[i_gene[1]+'_with_'+j_gene[1]] = 1
                                            

        print('unique pairs found %d, and count %d'%(len(list(negatome_unique_pair.keys())),count))









































































