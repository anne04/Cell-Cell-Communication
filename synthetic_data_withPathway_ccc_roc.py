import os
#import glob
import pandas as pd
#import shutil
import csv
import numpy as np
import sys
import altair as alt
from collections import defaultdict
import stlearn as st
import scanpy as sc
import qnorm
import scipy
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()

threshold_distance = 1.3 #2.3 #
k_nn = 8 # #5 = h
distance_measure = 'threshold_dist' # 'knn'  #<-----------
datatype = 'path_equally_spaced' #

'''
distance_measure = 'knn'  #'threshold_dist' # <-----------
datatype = 'pattern_high_density_grid' #'pattern_equally_spaced' #'mixture_of_distribution' #'equally_spaced' #'high_density_grid' 'uniform_normal' # <-----------'dt-pattern_high_density_grid_lrc1_cp20_lrp1_randp0_all_same_midrange_overlap'
'''
cell_percent = 20 # choose at random N% ligand cells
#neighbor_percent = 70
# lr_percent = 20 #40 #10
lr_count_percell = 1
receptor_connections = 'all_same' #'all_not_same'
gene_count = 4 #100 #20 #50 # and 25 pairs
rec_start = gene_count//2 #10 # 25
noise_add = 1  #2 #1
noise_percent = 30
random_active_percent = 0
active_type = 'random_overlap' #'highrange_overlap' #

def get_receptors(pattern_id, i, j, min_x, max_x, min_y, max_y, cell_neighborhood): #, dist_X, cell_id, cell_neighborhood):
    receptor_list = []
    if pattern_id == 1:
        '''	
        for n in cell_neighborhood:
            i_n = n[0]
            j_n = n[1]
            
            if i_n==i and j_n>j: 
                receptor_list.append([i,j_n])
            elif i_n==i and j_n<j:
                receptor_list.append([i,j_n])
            elif i_n>i and j_n==j: 
                receptor_list.append([i_n,j])
            elif i_n<i and j_n==j:
                receptor_list.append([i_n,j])
              
        '''
        receptor_list.append([i+1,j])
        receptor_list.append([i-1,j])
        receptor_list.append([i,j-1])       
        receptor_list.append([i,j+1]) 
        
        
        
        '''
        receptor_list.append([i+1,j])
        receptor_list.append([i,j-1])
	    '''
        #receptor_list.append([i+1,j-1])
        
    elif pattern_id == 2:
        receptor_list.append([i+1,j])
        receptor_list.append([i,j+1])
        #receptor_list.append([i+1,j+1])
        
    elif pattern_id == 3: 
        receptor_list.append([i,j-1])
        receptor_list.append([i-1,j])
        #receptor_list.append([i-1,j-1])
        
    elif pattern_id == 4: 
        receptor_list.append([i,j+1])
        receptor_list.append([i-1,j])
        #receptor_list.append([i-1,j+1])
        
    elif pattern_id == 5: 
        receptor_list.append([i+1,j])
        receptor_list.append([i-1,j])
        receptor_list.append([i,j-1])
        receptor_list.append([i,j+1])
        
    outside_boundary = 0   
    for index in receptor_list:
        if index[0] >= min_x and index[0] <= max_x and index[1] >= min_y and index[1] <= max_y:
            continue
        else:
            outside_boundary = 1
            break
    if outside_boundary == 1:
        return -1
    else :
        return receptor_list

def get_data(datatype):
    if datatype == 'path_equally_spaced':
        x_max = 50 #50 
        x_min = 0
        y_max = 50 #20 #30 
        y_min = 0
        temp_x = []
        temp_y = []
        index_dict = defaultdict(dict)
        i = x_min
        # row major order, bottom up
        k = 0
        while i < x_max:
          j = y_min
          while j < y_max:
              temp_x.append(i)
              temp_y.append(j)
              index_dict[i][j] = k
              k = k+1

              j = j + 1

          i = i + 1
        
        #0, 2, 4, ...24, 26, 28   
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        return temp_x, temp_y, 0
    
    elif datatype == 'pattern_high_density_grid':
        
        x_max = 100 #50 
        x_min = 0
        y_max = 80 #20
        y_min = 0
        temp_x = []
        temp_y = []
        i = x_min
        while i < x_max:
            j = y_min
            while j < y_max:
                temp_x.append(i)
                temp_y.append(j)
                j = j + 2
            i = i + 2

        #0, 2, 4, ...24, 26, 28 
        # high density
        region_list =  [[5, 30, 5, 25]] #[[20, 40, 3, 7], [40, 60, 12, 18]] #[60, 80, 1, 7] 	
        for region in region_list:
            x_max = region[1]
            x_min = region[0]
            y_min = region[2]
            y_max = region[3]
            i = x_min
            while i < x_max:
                j = y_min
                while j < y_max:
                    temp_x.append(i)
                    temp_y.append(j)
                    j = j + 2
                i = i + 2
        
        region_list.append([30, 65, 5, 15])       
        ccc_regions = []        
        for i in range (0, len(temp_x)):
            for region in region_list:
                x_max = region[1]
                x_min = region[0]
                y_min = region[2]
                y_max = region[3]
                if temp_x[i]>=x_min and temp_x[i]<=x_max and temp_y[i]>=y_min and temp_y[i]<=y_max:
                    ccc_regions.append(i)
                    
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        return temp_x, temp_y, ccc_regions
    
    elif datatype == 'pattern_mixture_of_distribution':
	
        datapoint_size = 2000
        x_max = 500
        x_min = 0
        y_max = 300
        y_min = 0
	
        a = x_min
        b = x_max
        #coord_x = np.random.randint(a, b, size=(datapoint_size))
        coord_x = (b - a) * np.random.random_sample(size=datapoint_size//2) + a

        a = y_min
        b = y_max
        coord_y = (b - a) * np.random.random_sample(size=datapoint_size//2) + a
        #coord_y = np.random.randint(a, b, size=(datapoint_size))

        temp_x = coord_x
        temp_y = coord_y
        region_list = [] 
        
        coord_x_t = np.random.normal(loc=200, scale=5, size=datapoint_size//8)
        coord_y_t = np.random.normal(loc=150, scale=5, size=datapoint_size//8)
        temp_x = np.concatenate((temp_x, coord_x_t))
        temp_y = np.concatenate((temp_y, coord_y_t))
        region_list.append([min(coord_x_t), max(coord_x_t), min(coord_y_t), max(coord_y_t)])
	
        coord_x_t = np.random.normal(loc=100, scale=10, size=datapoint_size//8)
        coord_y_t = np.random.normal(loc=100, scale=10, size=datapoint_size//8)
        temp_x = np.concatenate((temp_x, coord_x_t))
        temp_y = np.concatenate((temp_y, coord_y_t))
        #region_list.append([min(coord_x_t), max(coord_x_t), min(coord_y_t), max(coord_y_t)])
        
        coord_x_t = np.random.normal(loc=400,scale=15,size=datapoint_size//8)
        coord_y_t = np.random.normal(loc=100,scale=15,size=datapoint_size//8)
        temp_x = np.concatenate((temp_x, coord_x_t))
        temp_y = np.concatenate((temp_y, coord_y_t))
        #region_list.append([min(coord_x_t), max(coord_x_t), min(coord_y_t), max(coord_y_t)])
        
        coord_x_t = np.random.normal(loc=400,scale=20,size=datapoint_size//8)
        coord_y_t = np.random.normal(loc=200,scale=20,size=datapoint_size//8)
        temp_x = np.concatenate((temp_x, coord_x_t))
        temp_y = np.concatenate((temp_y, coord_y_t))
        region_list.append([min(coord_x_t), max(coord_x_t), min(coord_y_t), max(coord_y_t)])
        
        region_list.append([200, 350, 200, 300])
        
        discard_points = dict()
        for i in range (0, temp_x.shape[0]):
            if i not in discard_points:
                for j in range (i+1, temp_x.shape[0]):
                    if j not in discard_points:
                        if euclidean_distances(np.array([[temp_x[i],temp_y[i]]]), np.array([[temp_x[j],temp_y[j]]]))[0][0] < 1 :
                            print('i: %d and j: %d'%(i,j))
                            discard_points[j]=''

        coord_x = []
        coord_y = []
        for i in range (0, temp_x.shape[0]):
            if i not in discard_points:
                coord_x.append(temp_x[i])
                coord_y.append(temp_y[i])

        temp_x = coord_x
        temp_y = coord_y
        
        ccc_regions = []        
        for i in range (0, len(temp_x)):
            for region in region_list:
                x_max = region[1]
                x_min = region[0]
                y_min = region[2]
                y_max = region[3]
                if temp_x[i]>=x_min and temp_x[i]<=x_max and temp_y[i]>=y_min and temp_y[i]<=y_max:
                    ccc_regions.append(i)
                    
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
              
        return temp_x, temp_y, ccc_regions
          
        
        
        

################################# 
temp_x, temp_y, ccc_region = get_data(datatype)
#############################################
print(len(temp_x))
#plt.gca().set_aspect(1)	
#plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)
#save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'synthetic_spatial_plot_'+datatype+'.svg', dpi=400)
#plt.clf()


get_cell = defaultdict(dict)  
available_cells = []
for i in range (0, temp_x.shape[0]):
    get_cell[temp_x[i]][temp_y[i]] = i
    available_cells.append(i)
    
datapoint_size = temp_x.shape[0]
coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
distance_matrix = euclidean_distances(coordinates, coordinates)

     
########### weighted edge, based on neighborhood ##########
dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
cell_neighborhood = []
for i in range (0, datapoint_size):
    cell_neighborhood.append([])

for j in range(0, distance_matrix.shape[1]):
    max_value=np.max(distance_matrix[:,j])
    min_value=np.min(distance_matrix[:,j])
    for i in range(distance_matrix.shape[0]):
        dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)
        
    if distance_measure=='knn':
        list_indx = list(np.argsort(dist_X[:,j]))
        k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
        for i in range(0, distance_matrix.shape[0]):
            if i not in k_higher:
                dist_X[i,j] = 0 #-1
            else:
                cell_neighborhood[i].append([j, dist_X[i,j]])          
    else:
        for i in range(0, distance_matrix.shape[0]):
            # i to j: ligand is i 
            if distance_matrix[i,j] > threshold_distance: #i not in k_higher:
                dist_X[i,j] = 0 #-1
            else:
                cell_neighborhood[i].append([j, dist_X[i,j]])
		
	

for cell in range (0, len(cell_neighborhood)):
    cell_neighborhood_temp = cell_neighborhood[cell] 
    cell_neighborhood_temp = sorted(cell_neighborhood_temp, key = lambda x: x[1]) # sort based on distance
    cell_neighborhood[cell] = [] # to record the neighbor cells in that order
    for items in cell_neighborhood_temp:
        cell_neighborhood[cell].append(items[0])
####################################################################################            
# take gene_count normal distributions where each distribution has len(temp_x) datapoints.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html

cell_count = len(temp_x)
gene_distribution_active = np.zeros((gene_count, cell_count))
gene_distribution_inactive = np.zeros((gene_count, cell_count))
gene_distribution_noise = np.zeros((gene_count, cell_count))

start_loc = 4
for i in range (0, gene_count):
    gene_exp_list = np.random.normal(loc=start_loc+i,scale=.5,size=len(temp_x))
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i,:] =  gene_exp_list
    print('inactive: %g to %g'%(np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
    # np.min(gene_distribution_inactive[i,:])-3, scale=.5
    gene_exp_list = np.random.normal(loc=np.max(gene_distribution_inactive[i,:])+1, scale=.1, size=len(temp_x))   
    np.random.shuffle(gene_exp_list) 
    gene_distribution_active[i,:] = gene_exp_list  
    print('active: %g to %g'%(np.min(gene_distribution_active[i,:]),np.max(gene_distribution_active[i,:]) ))
    start_loc = np.max(gene_distribution_active[i,:])+3

#################################################
gene_ids = []
for i in range (0, gene_count):
    gene_ids.append(i) 

gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
#######################
lr_database = []
for i in range (0, rec_start):
    lr_database.append([i,rec_start+i])
    
ligand_dict_dataset = defaultdict(dict)
relation_id = 0
for i in range (0, len(lr_database)):
    ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i

ligand_list = list(ligand_dict_dataset.keys())   
    
#########################      	
cell_vs_gene = np.zeros((cell_count,gene_count))
# initially all are in inactive state
for i in range (0, gene_count):
    cell_vs_gene[:,i] = gene_distribution_inactive[i,:]

noise_cells = list(np.random.randint(0, cell_count, size=(cell_count*noise_percent)//100)) #“discrete uniform” distribution #ccc_region #
if noise_add == 1:
    gene_distribution_noise = np.random.normal(loc=0, scale=0.1, size = cell_vs_gene.shape[0])
    np.random.shuffle(gene_distribution_noise)	
    
    print('noise: %g to %g'%(np.min(gene_distribution_noise),np.max(gene_distribution_noise) ))
elif noise_add == 2:
    gene_distribution_noise = np.random.normal(loc=0, scale=.5, size = cell_vs_gene.shape[0])
    np.random.shuffle(gene_distribution_noise)	
    
    print('noise: %g to %g'%(np.min(gene_distribution_noise),np.max(gene_distribution_noise) ))
    

'''	
for i in range (0, len(noise_cells)):
    cell = noise_cells[i]
    cell_vs_gene[cell, :] = cell_vs_gene[cell, :] + gene_distribution_noise[i]
'''
# record true positive connections    
lig_rec_dict_TP = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size): 
    lig_rec_dict_TP.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict_TP[i].append([])   
        lig_rec_dict_TP[i][j] = []
	
# Pick the regions for Ligands

ligand_cells = list(np.random.randint(0, cell_count, size=(cell_count*cell_percent)//100)) #“discrete uniform” distribution #ccc_region #
set_ligand_cells = []
for i in ligand_cells:
    set_ligand_cells.append([temp_x[i], temp_y[i]])  


#lr_selected_list_allcell = list(np.random.randint(0, len(lr_database), size=len(ligand_cells))) 
#lr_selected_list_allcell = list(np.random.randint(0, len(lr_database), size=len(ligand_cells)*lr_count_percell))
k = 0
P_class = 0

all_used = dict()

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
    
for i in ligand_cells:
    # choose which L-R are working for this ligand i
    #lr_selected_list = [lr_selected_list_allcell[k]] 
    if i in all_used or cell_neighborhood[i][0] in all_used or cell_neighborhood[cell_neighborhood[i][0]][0] in all_used:
        print('skip')
        continue
    cell_vs_gene[i, :] = 0 # to set all other genes to 0, remember cell knee let us keep only active genes and others are set to 0.
    cell_vs_gene[cell_neighborhood[i][0], :] = 0 # to set all other genes to 0, remember cell knee let us keep only active genes and others are set to 0.
    cell_vs_gene[cell_neighborhood[cell_neighborhood[i][0]][0], :] = 0 # to set all other genes to 0, remember cell knee let us keep only active genes and others are set to 0.
    edge_list = []
    ##########################################    
    lr_i = 0
    ligand_gene = lr_database[lr_i][0]
    receptor_gene = lr_database[lr_i][1]
    cell_id = i
    cell_vs_gene[cell_id, ligand_gene] = gene_distribution_active[ligand_gene, cell_id]
    cell_id = cell_neighborhood[i][0]
    cell_vs_gene[cell_id, receptor_gene] = gene_distribution_active[receptor_gene, cell_id]
    edge_list.append([i, cell_neighborhood[i][0], ligand_gene, receptor_gene])
    #########################################
    lr_i = 1
    ligand_gene = lr_database[lr_i][0]
    receptor_gene = lr_database[lr_i][1]
    cell_id = cell_neighborhood[i][0]
    cell_vs_gene[cell_id, ligand_gene] = gene_distribution_active[ligand_gene, cell_id]
    cell_id = cell_neighborhood[cell_neighborhood[i][0]][0]
    cell_vs_gene[cell_id, receptor_gene] = gene_distribution_active[receptor_gene, cell_id]
    edge_list.append([cell_neighborhood[i][0], cell_neighborhood[cell_neighborhood[i][0]][0], ligand_gene, receptor_gene])
    ##########################################
    all_used[i] = ''
    all_used[cell_neighborhood[i][0]] = ''
    all_used[cell_neighborhood[cell_neighborhood[i][0]][0]] = ''
    ##########################################
    for c in [i, cell_neighborhood[i][0], cell_neighborhood[cell_neighborhood[i][0]][0]]:
        if c in noise_cells:
            for g in range (0, cell_vs_gene.shape[1]):
                if cell_vs_gene[c][g] != 0: ## CHECK ##
                    cell_vs_gene[c, g] = cell_vs_gene[c, g] + gene_distribution_noise[c]
                    
    for edge in edge_list:
        c1 = edge[0]
        c2 = edge[1]
        ligand_gene = edge[2]
        receptor_gene = edge[3]
        lig_rec_dict_TP[c1][c2].append(ligand_dict_dataset[ligand_gene][receptor_gene])
        P_class = P_class+1
        #########
        communication_score = cell_vs_gene[c1,ligand_gene] * cell_vs_gene[c2,receptor_gene] 
        communication_score = max(communication_score, 0)
        cells_ligand_vs_receptor[c1][c2].append([ligand_gene, receptor_gene, communication_score, ligand_dict_dataset[ligand_gene][receptor_gene]])              
        #########
                
# choose some random cells for activating without pattern


# take quantile normalization.
'''
temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  
adata_X = np.transpose(temp)  
cell_vs_gene = adata_X
'''


###############

# ready to go
################################################################################################
# do the usual things

for i in range (0, cell_vs_gene.shape[0]): # ligand                 
    for j in range (0, cell_vs_gene.shape[0]): # receptor
        if dist_X[i,j] <= 0: #distance_matrix[i,j] > threshold_distance:
            continue
        if i in all_used or j in all_used:
            continue
                
        for gene in ligand_list:
            rec_list = list(ligand_dict_dataset[gene].keys())
            for gene_rec in rec_list:   
                if i in noise_cells:
                    cell_vs_gene[i][gene_index[gene]] = cell_vs_gene[i][gene_index[gene]] + gene_distribution_noise[i]
                if j in noise_cells:
                    cell_vs_gene[j][gene_index[gene_rec]]  = cell_vs_gene[j][gene_index[gene_rec]]  + gene_distribution_noise[j]
              
                communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]     
                communication_score = max(communication_score, 0)
                if communication_score>0:
                    cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, ligand_dict_dataset[gene][gene_rec]])              
            

available_edges = []
for i in range (0, temp_x.shape[0]):
    for j in range (0, temp_x.shape[0]):
        if len(lig_rec_dict_TP[i][j])>0:
            continue
        if len(cells_ligand_vs_receptor[i][j]) > 0:
            lig = cells_ligand_vs_receptor[i][j][0][0]
            rec = lig = cells_ligand_vs_receptor[i][j][0][1]
            if cells_ligand_vs_receptor[i][j][0][2] < np.mean(gene_distribution_inactive[lig,:])*np.mean(gene_distribution_inactive[rec,:]): #distance_matrix[i,j] > threshold_distance:
                available_edges.append([i,j])  
                       
random_activation = []
random_activation_index = list(np.random.randint(0, len(available_edges), size=int(len(available_edges)*(random_active_percent))//100))
random_activation_lr = list(np.random.randint(0, len(lr_database), size= len(random_activation_index)*lr_count_percell)) #“discrete uniform” distribution
k = 0
p=0
for index in random_activation_index:
    i = available_edges[index][0]
    j = available_edges[index][1]
    
    lr_i = random_activation_lr[p]
    p = p + 1
                
    ligand_gene = lr_database[lr_i][0]
    cell_vs_gene[i,ligand_gene] = cell_vs_gene[i,ligand_gene] + gene_distribution_noise[ligand_gene, i] 
    
    receptor_gene = lr_database[lr_i][1]
    cell_vs_gene[j,receptor_gene] = cell_vs_gene[j,receptor_gene] + gene_distribution_noise[receptor_gene, j] 
    
    if cells_ligand_vs_receptor[i][j][lr_i][0] == ligand_gene and cells_ligand_vs_receptor[i][j][lr_i][1] == receptor_gene:
        cells_ligand_vs_receptor[i][j][lr_i][2] = cell_vs_gene[i,ligand_gene]*cell_vs_gene[j,receptor_gene]
        
    
    
    random_activation.append([i,j])
    
        
    #k = k + lr_count_percell
 
  
    

ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((20))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if dist_X[i,j] > 0: #distance_matrix[i][j] <= threshold_distance: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    count_edge = count_edge + 1
                    count_local = count_local + 1
                    #print(count_edge)                      
                    mean_ccc = cells_ligand_vs_receptor[i][j][k][2]  #*dist_X[i,j]
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    lig_rec.append(cells_ligand_vs_receptor[i][j][k][3])
                if max_local < count_local:
                    max_local = count_local
            '''       
            else: #elif i in all_used and j in all_used:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
        	'''
                
            ''' '''
            #local_list[count_local] = local_list[count_local] + 1


		
print('len row col %d'%len(row_col))
print('max local %d'%max_local) 
print('random_activation %d'%len(random_activation_index))
print('ligand_cells %d'%len(ligand_cells))
print('P_class %d'%P_class) 

options = 'dt-'+datatype+'_lrc'+str(len(lr_database))+'_cp'+str(cell_percent)+'_lrp'+str(lr_count_percell)+'_randp'+str(random_active_percent)+'_'+receptor_connections+'_noise'+str(noise_percent)#'_close'
if noise_add == 1:
    options = options + '_lowNoise'
if noise_add == 2:
    options = options + '_heavyNoise'

total_cells = len(cells_ligand_vs_receptor)

options = options+ '_' + active_type + '_' + distance_measure  + '_cellCount' + str(total_cells)

lig_rec_dict_TP_temp = defaultdict(dict)
for i in range (0, len(lig_rec_dict_TP)):
    for j in range (0, len(lig_rec_dict_TP)):
        if len(lig_rec_dict_TP[i][j]) > 0:
            lig_rec_dict_TP_temp[i][j] = []
            
for i in range (0, len(lig_rec_dict_TP)):
    for j in range (0, len(lig_rec_dict_TP)):
        if len(lig_rec_dict_TP[i][j]) > 0:
            for k in range (0, len(lig_rec_dict_TP[i][j])):
               lig_rec_dict_TP_temp[i][j].append(lig_rec_dict_TP[i][j][k]) 
            
lig_rec_dict_TP = lig_rec_dict_TP_temp

#options = options+ '_' + 'wFeature'
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options +'_'+'quantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lr_database, lig_rec_dict_TP], fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'quantileTransformed_communication_scores', 'wb') as fp: #b, b_1, a
    pickle.dump(cells_ligand_vs_receptor, fp) #a - [0:5]
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'quantileTransformed', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)
    
'''   
'''
options = options + '_' + active_type + '_' + 'noisy'    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'notQuantileTransformed', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'notQuantileTransformed_communication_scores', 'wb') as fp: #b, b_1, a
#    pickle.dump(cells_ligand_vs_receptor, fp) #a - [0:5]
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options +'_'+'notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec], fp)
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options +'_'+'notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([lr_database, lig_rec_dict_TP, random_activation], fp)
        
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'wb') as fp:
    pickle.dump([temp_x, temp_y, ccc_region], fp)
   
 '''  
###############
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'notQuantileTransformed', 'rb') as fp:
    cell_vs_gene = pickle.load(fp)
'''
###########################################################

'''
2000
gene 0, min: 0, max:13.7769 
gene 1, min: 0, max:13.4894 
gene 2, min: 0, max:13.7656 
gene 3, min: 0, max:13.8944 
gene 0, min: 22.5465, max:37.0469 
gene 1, min: 28.4818, max:41.517 
gene 2, min: 34.072, max:45.9164 
gene 3, min: 38.2734, max:53.4325 
len row col 177016
count local 2
'''
###############################################Visualization starts###################################################################################################
# 'dt-mixture_of_distribution_lrc5_cp10_np70_lrp40_all_same'
# 'dt-high_density_grid_lrc5_cp10_np70_lrp40_all_same'
# 'dt-high_density_grid_lrc5_cp10_np70_lrp40_all_same_close'
# 'dt-high_density_grid_lrc5_cp10_np70_lrp40_all_same_noisy'
# 'dt-equally_spaced_lrc5_cp10_np70_lrp40_all_same'
# 'dt-high_density_grid_lrc5_cp10_np70_lrp40_all_same_close_noisy'
# 'dt-high_density_grid_lrc50_cp10_np70_lrp40_all_same_close_noisy'
# 'dt-high_density_grid_lrc5_cp10_np70_lrp40_all_same_close_heavy_noisy'
# 'dt-pattern_equally_spaced_lrc5_cp50_lrp40_randp30_all_same'
# 'dt-pattern_equally_spaced_lrc5_cp50_lrp20_randp5_all_same'
# 'dt-pattern_equally_spaced_lrc5_cp80_lrp3_randp0_all_same'
# 'dt-pattern_equally_spaced_lrc5_cp80_lrp20_randp0_all_same' 
# 'dt-pattern_equally_spaced_lrc1_cp70_lrp1_randp0_all_same'
# 'dt-pattern_equally_spaced_lrc1_cp90_lrp1_randp0_all_same' --withFeature_pattern_4_attention, model_4_pattern_attention
# 'dt-pattern_equally_spaced_lrc1_cp10_lrp1_randp0_all_same'
# 'dt-pattern_equally_spaced_lrc1_cp10_lrp1_randp0_all_same_broad_active'
# 'dt-pattern_equally_spaced_lrc1_cp10_lrp1_randp0_all_same_overlapped_lowscale'
# 'dt-pattern_equally_spaced_lrc5_cp50_lrp1_randp0_all_same_differentLRs'
# 'dt-pattern_equally_spaced_lrc4_cp50_lrp1_randp0_all_sameoverlapped_highertail'
options = 'dt-'+datatype+'_lrc'+str(25)+'_cp'+str(cell_percent)+'_np'+str(neighbor_percent)+'_lrp'+str(lr_percent)+'_'+receptor_connections

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options  +'_xny', 'rb') as fp: #datatype
    temp_x, temp_y , ccc_region = pickle.load(fp) #

datapoint_size = temp_x.shape[0]

coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

#####################################, random_activation
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options +'_'+'quantileTransformed', 'rb') as fp:  # at least one of lig or rec has exp > respective knee point          
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options +'_'+'notQuantileTransformed', 'rb') as fp:  # at least one of lig or rec has exp > respective knee point          
    row_col, edge_weight, lig_rec  = pickle.load(fp)  #, lr_database, lig_rec_dict_TP, random_activation
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options +'_'+'notQuantileTransformed', 'rb') as fp:  # at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)
        
datapoint_size = temp_x.shape[0]              
total_type = np.zeros((len(lr_database)))
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
            for k in range (0, len(lig_rec_dict_TP[i][j])):
               total_type[lig_rec_dict_TP[i][j][k]] = total_type[lig_rec_dict_TP[i][j][k]] + 1
               
positive_class = np.sum(total_type)
negative_class = len(row_col) - positive_class           
############# draw the points which are participating in positive classes  ######################
ccc_index_dict = dict()               
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
            #if 1 in lig_rec_dict_TP[i][j]:                
            ccc_index_dict[i] = ''
            ccc_index_dict[j] = ''               

######################################	

attention_scores = []
lig_rec_dict = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    lig_rec_dict.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []
        
#attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    if 1==1: #lig_rec[index] in lig_rec_dict_TP[i][j]: # and lig_rec[index]==1:  
        lig_rec_dict[i][j].append(lig_rec[index])
        #attention_scores[i][j] = edge_weight[index][1]
        #attention_scores[i][j].append(edge_weight[index][1])
        #distribution.append(edge_weight[index][1])    
        attention_scores[i][j].append(edge_weight[index][1]*edge_weight[index][0])
        distribution.append(edge_weight[index][1]*edge_weight[index][0])

percentage_value = 100
while percentage_value > 0:
    percentage_value = percentage_value - 10
#for percentage_value in [79, 85, 90, 93, 95, 97]:
    existing_lig_rec_dict = []
    datapoint_size = temp_x.shape[0]
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []

    
    threshold_down =  np.percentile(sorted(distribution), percentage_value)
    threshold_up =  np.percentile(sorted(distribution), 100)
    connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
    rec_dict = defaultdict(dict)
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if i==j: 
                continue
            atn_score_list = attention_scores[i][j]
            #print(len(atn_score_list))
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])


    #############
    positive_class = 0  
    negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue
                
            if len(lig_rec_dict[i][j])>0:
                for k in lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        positive_class = positive_class + 1
                        if k in existing_lig_rec_dict[i][j]:
                            confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                        else:
                            confusion_matrix[0][1] = confusion_matrix[0][1] + 1                 
                    else:
                        negative_class = negative_class + 1
                        if k in existing_lig_rec_dict[i][j]:
                            confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                        else:
                            confusion_matrix[1][1] = confusion_matrix[1][1] + 1      
                            
    print('%d, %g, %g'%(percentage_value, (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    


'''
ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 98)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for j in range (0, datapoint_size):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        for k in range (0, len(atn_score_list)):   
            if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                connecting_edges[i][j] = 1
                #lig_rec_dict_filtered[i][j].append(lig_rec_dict[i][j][k][1])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
'''
################

########withFeature withFeature_
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_pathway_random_overlapped_lowNoise_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

l=3 #2 ## 
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    distribution.append(X_attention_bundle[l][index][0])


max_value = np.max(distribution)
	
#attention_scores = np.zeros((2000,2000))
tweak = 0
distribution = []
attention_scores = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []

for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index] 
    #if i>= temp_x.shape[0] or  j>= temp_x.shape[0]:
    #    continue
    ###################################
    
    if tweak == 1:         
        attention_scores[i][j].append(max_value+(X_attention_bundle[l][index][0]*(-1)) ) #X_attention_bundle[2][index][0]
        distribution.append(max_value+(X_attention_bundle[l][index][0]*(-1)) )
    else:
        attention_scores[i][j].append(X_attention_bundle[l][index][0]) 
        distribution.append(X_attention_bundle[l][index][0])
#######################

percentage_value = 100
while percentage_value > 0:
    percentage_value = percentage_value - 10
#for percentage_value in [79, 85, 90, 93, 95, 97]:
    existing_lig_rec_dict = []
    datapoint_size = temp_x.shape[0]
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []

    ccc_index_dict = dict()
    threshold_down =  np.percentile(sorted(distribution), percentage_value)
    threshold_up =  np.percentile(sorted(distribution), 100)
    connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
    rec_dict = defaultdict(dict)
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if i==j: 
                continue
            atn_score_list = attention_scores[i][j]
            #print(len(atn_score_list))
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])


    #############
    positive_class = 0  
    negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue
                
            if len(lig_rec_dict[i][j])>0:
                for k in lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        positive_class = positive_class + 1
                        if k in existing_lig_rec_dict[i][j]:
                            confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                        else:
                            confusion_matrix[0][1] = confusion_matrix[0][1] + 1                 
                    else:
                        negative_class = negative_class + 1
                        if k in existing_lig_rec_dict[i][j]:
                            confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                        else:
                            confusion_matrix[1][1] = confusion_matrix[1][1] + 1      
                            
    print('%d, %g, %g'%(percentage_value, (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    
                     
                        
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

datapoint_label = []
node_list = []
for i in range (0, temp_x.shape[0]):
    if count_points_component[labels[i]]>1:
        datapoint_label.append(2) #
        #if coordinates[i][0] <100 and (coordinates[i][1]>150 and coordinates[i][1]<250):
            #print('%d'%i)
            #node_list.append(i)
        #datapoint_label.append(index_dict[labels[i]])
    else:
        datapoint_label.append(0)
	
#############
'''
datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if i in ccc_index_dict:
        datapoint_label.append(2)
    else:
        datapoint_label.append(0)
        
for i in random_activation:
    datapoint_label[i[0]] = 1 
    datapoint_label[i[1]] = 1 

'''
########
plt.gca().set_aspect(1)	

number = 20
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

number = 20
cmap = plt.get_cmap('tab20b')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 8
cmap = plt.get_cmap('Set2')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 12
cmap = plt.get_cmap('Set3')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2
               
plt.gca().set_aspect(1)	       
id_label = [0,2]
for j in id_label:
#for j in range (0, id_label+1):
    x_index=[]
    y_index=[]
    #fillstyles_type = []
    for i in range (0, temp_x.shape[0]):
        if datapoint_label[i] == j:
            x_index.append(temp_x[i])
            y_index.append(temp_y[i])
    #print(len(x_index))
            
            ##############
    plt.scatter(x=x_index, y=y_index, label=j, color=colors[j], s=4)   
    
plt.legend(fontsize=4,loc='upper right')


save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 
plt.hist(distribution, color = 'blue',
         bins = int(len(distribution)/5))
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
####################
ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(temp_x)):    
    ids.append(i)
    x_index.append(temp_x[i]*100)
    y_index.append(temp_y[i]*100)    
    colors_point.append(colors[datapoint_label[i]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)


from pyvis.network import Network
import networkx as nx
import matplotlib#.colors.rgb2hex as rgb2hex
    
g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(temp_x)):
    marker_size = 'circle'
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network(directed=True) #"500px", "500px",
nt.from_nx(g)
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):
            #if attention_scores[i][j][k] >= threshold_down:
                #print('hello')
                title_str =  ""+str(lig_rec_dict[i][j][k])+", "+str(attention_scores[i][j][k])
                nt.add_edge(int(i), int(j), title=title_str) #, value=np.float64(attention_scores[i][j][k])) #,width=, arrowsize=int(20),  arrowstyle='fancy'

nt.show('mygraph.html')


#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html