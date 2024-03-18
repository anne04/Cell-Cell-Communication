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
import altair as alt
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
import gc
import copy
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset') # 
    parser.add_argument( '--data_from', type=str, default='/cluster/projects/schwartzgroup/fatema/?/outs/' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.')
    parser.add_argument( '--filter_min_cell', type=int, default=1 , help='Minimum number of cells for gene filtering') 
    parser.add_argument( '--tissue_position_file', type=str, default='None', help='If your --data_from argument points to a *.mtx file instead of Space Ranger, then please provide the path to tissue position file.')
    parser.add_argument( '--output_name', type=str, default='NEST_figures_output/', help='Output file name prefix according to user\'s choice')
    args = parser.parse_args()

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
        cell_vs_gene = sparse.csr_matrix.toarray(adata_h5.X)   

    else:
        
        temp = sc.read_10x_mtx(args.data_from)
        print('*.mtx file read done')
        gene_count_before = len(list(temp.var_names) )
        sc.pp.filter_genes(temp, min_cells=args.filter_min_cell)
        gene_count_after = len(list(temp.var_names) )
        print('Gene filtering done. Number of genes reduced from %d to %d'%(gene_count_before, gene_count_after))
        gene_ids = list(temp.var_names) 
        cell_barcode = np.array(temp.obs.index)
        cell_vs_gene = sparse.csr_matrix.toarray(temp.X)
    
        
        # now read the tissue position file. It has the format:     
        df = pd.read_csv(args.tissue_position_file, sep=",", header=None)   
        tissue_position = df.values
        barcode_vs_xy = dict() # record the x and y coordinates for each spot/cell
        for i in range (0, tissue_position.shape[0]):
            barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][5], tissue_position[i][4]] #for some weird reason, in the .h5 format for LUAD sample, the x and y are swapped
        
        coordinates = np.zeros((cell_barcode.shape[0], 2)) # insert the coordinates in the order of cell_barcodes
        for i in range (0, cell_barcode.shape[0]):
            coordinates[i,0] = barcode_vs_xy[cell_barcode[i]][0]
            coordinates[i,1] = barcode_vs_xy[cell_barcode[i]][1]
    

    ##################### make metadata: barcode_info ###################################
    i=0
    barcode_info=[]
    for cell_code in cell_barcode:
        barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1], 0]) # last entry will hold the component number later
        i=i+1


    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th', 'rb') as fp:  #b, a:[0:5]   
        row_col, edge_weight, lig_rec = pickle.load(fp)
    
    
    attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
    distribution = []
    for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        attention_scores[i][j] = edge_weight[index][1]
        distribution.append(attention_scores[i][j])
        
        
    
    threshold =  np.percentile(sorted(distribution), 95)
    connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
    
    for j in range (0, attention_scores.shape[1]):
        #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
        for i in range (0, attention_scores.shape[0]):
            if attention_scores[i][j] > threshold: #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
                connecting_edges[i][j] = 1
                
    
    
    graph = csr_matrix(connecting_edges)
    n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
    print('number of component %d'%n_components)
    
    count_points_component = np.zeros((n_components))
    for i in range (0, len(labels)):
         count_points_component[labels[i]] = count_points_component[labels[i]] + 1
               
    print(count_points_component)
    
    id_label = 0  
    index_dict = dict()
    for i in range (0, count_points_component.shape[0]):
        if count_points_component[i]>1:
            id_label = id_label+1
            index_dict[i] = id_label
    print(id_label)
            

    for i in range (0, len(barcode_info)):
        barcode_info[i][3] = 0 # initially all are zero
        if count_points_component[labels[i]] > 1:
            barcode_info[i][3] = index_dict[labels[i]]
        
           
    
    ########
    number = 20
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    
    
    plt.clf()
    x_index=[]
    y_index=[]
    marker_size = []   
    spot_color = []
    label = []
    for i in range (0, len(barcode_info)):
        x_index.append(barcode_info[i][1])
        y_index.append(-barcode_info[i][2])
        marker_size.append(matplotlib.markers.MarkerStyle(marker='o', fillstyle=filltype))
        spot_color.append(colors[0])
        label.append(-1)
        
    cell_count_cluster=np.zeros((labels.shape[0]))
    filltype='full'
    for j in range (0, n_components):
        label_i = j
        
        for i in range (0, len(barcode_info)):
            if barcode_info[i][3] == label_i:
                if label_i>1:
                    label_i=1
                label[i] = label_i
                spot_color[i]=colors[label_i]
                
   
    plt.scatter(x=x_index, y=y_index, label = label, color=spot_color, s=15) 
    save_path = '/cluster/home/t116508uhn/64630/'
    plt.savefig(save_path+'plot_naive_pdac_64630.svg', dpi=400)
