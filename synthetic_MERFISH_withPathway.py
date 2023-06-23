import os
#import glob
import pandas as pd
#import shutil
import copy
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
from kneed import KneeLocator


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()

threshold_distance = 2 #2 = path equally spaced
k_nn = 10 # #5 = h
distance_measure = 'knn'  #'threshold_dist' # <-----------
datatype = 'path_uniform_distribution' #'path_equally_spaced' #

cell_percent = 100 # choose at random N% ligand cells
#neighbor_percent = 70
# lr_percent = 20 #40 #10
#lr_count_percell = 1
#receptor_connections = 'all_same' #'all_not_same'

total_gene = 300
lr_gene_count = 42*2 #8 #100 #20 #100 #20 #50 # and 25 pairs
rec_start = lr_gene_count//2 # 25
ligand_gene_list = np.arange(0, lr_gene_count//2)
receptor_gene_list = np.arange(lr_gene_count//2, lr_gene_count)
# which ligand to which receptor
np.random.shuffle(ligand_gene_list) 
np.random.shuffle(receptor_gene_list) 
gene_group = [] #[[[],[]], [[],[]] ,[[],[]] ,[[],[]] ,[[],[]]] # [3*3]*15 = 120 lr pairs
gene_group_count = 14
for i in range (0, gene_group_count):
	gene_group.append([list(ligand_gene_list[i*3:(i+1)*3]),list(receptor_gene_list[i*3:(i+1)*3])])
    


lr_database = []
lr_database_index = defaultdict(dict) # [ligand][recptor] = index of that pair in the lr_database
for i in range (0, len(gene_group)):
    lr_group = gene_group[i]
    for j in lr_group[0]: # all of these ligands 
        for k in lr_group[1]: # match with all of these receptors
            lr_database.append([j, k])
            lr_database_index[j][k] = len(lr_database)-1 # indexing starts from 0

pattern_list = []
pattern_count = len(gene_group)//2
j = 0
for i in range (0, pattern_count):
    pattern_list.append([j, j+1])
    j = j+2
#pattern_list = [[0, 1], [2, 3], [4]] # e.g., first pattern: a --> b: group 0; b --> c: group 1.    

non_lr_genes = total_gene - lr_gene_count

noise_add = 0  #2 #1
noise_percent = 0
random_active_percent = 0
#active_type = 'random_overlap' #'highrange_overlap' #


def get_data(datatype):
    if datatype == 'path_uniform_distribution':	
        datapoint_size = 5000
        x_max = 600
        x_min = 0
        y_max = 600
        y_min = 0
	
        a = x_min
        b = x_max
        #coord_x = np.random.randint(a, b, size=(datapoint_size))
        coord_x = (b - a) * np.random.random_sample(size=datapoint_size) + a

        a = y_min
        b = y_max
        coord_y = (b - a) * np.random.random_sample(size=datapoint_size) + a
        #coord_y = np.random.randint(a, b, size=(datapoint_size))

        temp_x = coord_x
        temp_y = coord_y
        region_list = [] 
        '''
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
        '''
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
print("cell count %d"%len(temp_x))

options = 'dt-'+datatype+'_lrc'+str(len(lr_database))+'_noise'+str(noise_percent)#'_close'
if noise_add == 1:
    options = options + '_lowNoise'
if noise_add == 2:
    options = options + '_heavyNoise'


datapoint_size = temp_x.shape[0]
options = options+ '_' + distance_measure  + '_cellCount' + str(datapoint_size)

options = options + '_g'
options = options + '_3dim'
#options = options + '_scaled'
# options = 'dt-path_mixture_of_distribution_lrc126_noise0_knn_cellCount4565_g_3dim'
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'wb') as fp:
    pickle.dump([temp_x, temp_y, ccc_region], fp)

fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'rb')
temp_x, temp_y, ccc_region = pickle.load(fp)



#plt.gca().set_aspect(1)	
#plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)
#save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'synthetic_spatial_plot_'+datatype+'.svg', dpi=400)
#plt.clf()

get_cell = defaultdict(dict)  #[x_index][y_index] = cell_id
available_cells = []
for i in range (0, temp_x.shape[0]):
    get_cell[temp_x[i]][temp_y[i]] = i 
    available_cells.append(i)
    

coordinates = np.zeros((datapoint_size,2))
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
    cell_neighborhood_temp = sorted(cell_neighborhood_temp, key = lambda x: x[1], reverse=True) 
	# sort based on distance: large to small because that is how it is weighted. Closer neighbor has higher weight
    
    cell_neighborhood[cell] = [] # to record the neighbor cells in that order
    for items in cell_neighborhood_temp:
        cell_neighborhood[cell].append(items[0])
    #np.random.shuffle(cell_neighborhood[cell]) 
####################################################################################            
# take lr_gene_count normal distributions where each distribution has len(temp_x) datapoints.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html

max_neighbor = 0
for i in range (0, len(cell_neighborhood)):
    if len(cell_neighborhood[i])>max_neighbor:
        max_neighbor = len(cell_neighborhood[i])
        
print('max number of neighbors: %d'%max_neighbor)

for attempt in range (0, 1):
    print("attempt %d"%attempt)
    cell_count = datapoint_size
    gene_distribution_active = np.zeros((total_gene, cell_count))
    gene_distribution_inactive = np.zeros((total_gene, cell_count))
    gene_distribution_noise = np.zeros((total_gene, cell_count))
    
    # innitialize gene expression for the lr genes 
    start_loc = 15 # making it higher than non-lr gene will give more FP to confuse the naive model
    i = 0
    for gene_index in ligand_gene_list: 
        gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=cell_count)
        for cell_index in range (0, gene_exp_list.shape[0]): # make the min exp = 0
            if gene_exp_list[cell_index]<0:
                gene_exp_list[cell_index] = 0
        np.random.shuffle(gene_exp_list) 
        gene_distribution_inactive[gene_index,:] =  gene_exp_list
        #print('%d: inactive: %g to %g'%(gene_index, np.min(gene_distribution_inactive[gene_index,:]),np.max(gene_distribution_inactive[gene_index,:]) ))
        i = i+1
    
    for gene_index in receptor_gene_list: 
        gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=cell_count)
        for cell_index in range (0, gene_exp_list.shape[0]): # make the min exp = 0
            if gene_exp_list[cell_index]<0:
                gene_exp_list[cell_index] = 0    
        np.random.shuffle(gene_exp_list) 
        gene_distribution_inactive[gene_index,:] =  gene_exp_list
        #print('%d: inactive: %g to %g'%(gene_index, np.min(gene_distribution_inactive[gene_index,:]),np.max(gene_distribution_inactive[gene_index,:]) ))
        i = i+1
        
    start_loc = 10
    non_lr_gene_start = lr_gene_count
    for i in range (non_lr_gene_start, total_gene):
        gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=3,size=cell_count)
        for cell_index in range (0, gene_exp_list.shape[0]): # make the min exp = 0
            if gene_exp_list[cell_index]<0:
                gene_exp_list[cell_index] = 0
        
        np.random.shuffle(gene_exp_list) 
        gene_distribution_inactive[i,:] =  gene_exp_list
        #print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
     
    ################# define active state of the ligand-receptor genes #############################
    start_loc = np.max(gene_distribution_inactive) + 30 # 28
    # does not matter in naive model performance since you are turning off all other genes in the active spots and also neighboring spots (within threshold distance)
    # bringing it closer to the exp of non active spot's exp will need higher threshold at the end to get all the TP edges in the input graph. As a result you will 
    # end up selecting too many edges where most of them lie in the lower end of the distribution. 
    
    rec_gene = lr_gene_count//2
    scale_active_distribution = 1 #0.01
    i = 0
    for gene_index in ligand_gene_list:
        gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=cell_count) #
        np.random.shuffle(gene_exp_list) 
        gene_distribution_active[gene_index,:] =  gene_exp_list
        #print('%d: active: %g to %g'%(gene_index, np.min(gene_distribution_active[gene_index,:]),np.max(gene_distribution_active[gene_index,:]) ))
        i = i+1
        
    for gene_index in receptor_gene_list:
        gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=cell_count) #
        np.random.shuffle(gene_exp_list) 
        gene_distribution_active[gene_index,:] =  gene_exp_list
        #print('%d: active: %g to %g'%(gene_index, np.min(gene_distribution_active[gene_index,:]),np.max(gene_distribution_active[gene_index,:]) ))
        i = i+1  
        
    #################################################
    
    print('min lr_gene_exp %g'%np.min(gene_distribution_inactive))
    print('max lr_gene_exp %g'%np.max(gene_distribution_inactive))
    
    gene_ids = []
    for i in range (0, lr_gene_count):
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
    '''
    lr_database = []
    
    for i in range (0, rec_start):
        lr_database.append([i,rec_start+i])
    '''
    
    ligand_dict_dataset = defaultdict(dict)
    for i in range (0, len(lr_database)):
        ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i
    
    ligand_list = list(ligand_dict_dataset.keys())   
        
    #########################      	
    cell_vs_gene = np.zeros((cell_count, total_gene))
    # initially all are in inactive state
    for i in range (0, total_gene):
        cell_vs_gene[:,i] = gene_distribution_inactive[i,:]
    
    '''
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
    '''	
    for i in range (0, len(noise_cells)):
        cell = noise_cells[i]
        cell_vs_gene[cell, :] = cell_vs_gene[cell, :] + gene_distribution_noise[i]
    '''
    # record true positive connections    
    lig_rec_dict_TP = []
    for i in range (0, datapoint_size): 
        lig_rec_dict_TP.append([])  
        for j in range (0, datapoint_size):	
            lig_rec_dict_TP[i].append([])   
            lig_rec_dict_TP[i][j] = []
            
    P_class = 0
    ########################################
    
    pattern_used = dict() # record the cells which follow any of the patterns / are turned active
    
    pattern_used_list = []
    for i in range (0, pattern_count):
        pattern_used_list.append(dict())
    
    #pattern_used_0 = dict() # record the cells which follow pattern 0
    #pattern_used_1 = dict() # record the cells which follow pattern 1
    #pattern_used_2 = dict() # record the cells which follow pattern 2
    # pattern_used_3 = dict()
    active_spot = dict()
    # Pick the regions for Ligands
    '''
    cells_ligand_vs_receptor = []
    for i in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor.append([])
    
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[0]):
            cells_ligand_vs_receptor[i].append([])
            cells_ligand_vs_receptor[i][j] = []
    '''
    min_lr_gene_exp = np.min(gene_distribution_inactive)
    p_dist = []
    for pattern_type_index in range (0, pattern_count): 
        discard_cells = list(pattern_used.keys()) + list(active_spot.keys())    
        ligand_cells = list(set(np.arange(cell_count)) - set(discard_cells))
        ligand_cells = ligand_cells[0: min(len(ligand_cells), cell_count//(pattern_count*10))] # 10.  1/N th of the all cells are following this pattern, where, N = total patterns
        np.random.shuffle(ligand_cells)
        print("pattern_type_index %d, ligand_cell count %d"%(pattern_type_index, len(ligand_cells)))
        print(ligand_cells[0:10])
    
        xy_ligand_cells = []
        for i in ligand_cells:
            xy_ligand_cells.append([temp_x[i], temp_y[i]]) 
            
        k= -1
        for i in ligand_cells:
            cell_of_interest = []
            # choose which L-R are working for this ligand i
            k = k + 1 
            
            a_cell = i
            b_cell = cell_neighborhood[a_cell][len(cell_neighborhood[a_cell])-1]  
            cell_of_interest.append(a_cell)
            cell_of_interest.append(b_cell)
            
            #if pattern_type_index != 2:        
            if cell_neighborhood[b_cell][len(cell_neighborhood[b_cell])-1]!=a_cell:
                c_cell = cell_neighborhood[b_cell][len(cell_neighborhood[b_cell])-1]
            else:
                c_cell = cell_neighborhood[b_cell][len(cell_neighborhood[b_cell])-2]
            cell_of_interest.append(c_cell)
            
            edge_list = []
            if a_cell in pattern_used_list[pattern_type_index] or b_cell in pattern_used_list[pattern_type_index] or c_cell in pattern_used_list[pattern_type_index]: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in pattern_used:
            #print('skip')
                continue   
            if a_cell in pattern_used or b_cell in pattern_used or c_cell in pattern_used: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in pattern_used:
            #print('skip')
                continue             
            '''
            if pattern_type_index == 0:
                if a_cell in pattern_used_0 or b_cell in pattern_used_0 or c_cell in pattern_used_0: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in pattern_used:
                #print('skip')
                    continue          
            elif pattern_type_index == 1:
                if a_cell in pattern_used_1 or b_cell in pattern_used_1 or c_cell in pattern_used_1: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in pattern_used:
                #print('skip')
                    continue        
            elif pattern_type_index == 2:
                if a_cell in pattern_used_2 or b_cell in pattern_used_2: # or c_cell in pattern_used_2: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in pattern_used:
                #print('skip')
                    continue            
            '''
                    
            a = pattern_list[pattern_type_index][0] # 0
            #if pattern_type_index != 2:    
            b = pattern_list[pattern_type_index][1] # 1
            ##########################################   
            # turn on all the genes that are in pattern 'a'        
            ligand_group_a_cell = gene_group[a][0]
            cell_id = a_cell
            for gene in ligand_group_a_cell:
                cell_vs_gene[cell_id, gene] = gene_distribution_active[gene, cell_id]
            
            
            cell_id = b_cell
            receptor_group_b_cell = gene_group[a][1]
            for gene in receptor_group_b_cell:
                cell_vs_gene[cell_id, gene] = gene_distribution_active[gene, cell_id]
    
            for ligand_gene in ligand_group_a_cell:
                for receptor_gene in receptor_group_b_cell:
                    edge_list.append([a_cell, b_cell, ligand_gene, receptor_gene])
    
            
            #if pattern_type_index!=2:
            ligand_group_b_cell = gene_group[b][0]
            for gene in ligand_group_b_cell:
                cell_vs_gene[cell_id, gene] = gene_distribution_active[gene, cell_id]
                
            cell_id = c_cell
            receptor_group_c_cell = gene_group[b][1]
            for gene in receptor_group_c_cell:
                cell_vs_gene[cell_id, gene] = gene_distribution_active[gene, cell_id]
       
            for ligand_gene in ligand_group_b_cell:
                for receptor_gene in receptor_group_c_cell:
                    edge_list.append([b_cell, c_cell, ligand_gene, receptor_gene])
    
            #################
            # [0, 1, 2, 3,  8, 9, 10, 11]
            # a_cell has only 0, 2 active. b_cell has 8, 10, 1, 3 active. c_cell has 9, 11 active.  
    
            for cell in cell_of_interest:
                active_spot[cell] = ''
    
            ###############################################################
            gene_off_list = receptor_group_b_cell + ligand_group_b_cell 
            if pattern_type_index!=2:
                gene_off_list = gene_off_list + receptor_group_c_cell
            for gene in gene_off_list:
                cell_vs_gene[a_cell, gene] = min_lr_gene_exp #-10
                
            for cell in cell_neighborhood[a_cell]:            
                if cell in cell_of_interest:
                    continue
                if cell in active_spot:
                    continue
                    
                pattern_used[cell]=''
                for gene in gene_off_list:
                    cell_vs_gene[cell, gene] = min_lr_gene_exp #-10   
                
            
            #################################################################
            gene_off_list = ligand_group_a_cell 
            #if pattern_type_index!=2:
            gene_off_list = gene_off_list + receptor_group_c_cell        
            
            for gene in gene_off_list:
                cell_vs_gene[b_cell, gene] = min_lr_gene_exp #-10
                
            for cell in cell_neighborhood[b_cell]:            
                if cell in cell_of_interest:
                    continue
                if cell in active_spot:
                    continue
                    
                pattern_used[cell]=''
                for gene in gene_off_list:
                    cell_vs_gene[cell, gene] = min_lr_gene_exp #-10   
                       
            #################################################################
            #if pattern_type_index!=2:
            gene_off_list = ligand_group_a_cell  + receptor_group_b_cell + ligand_group_b_cell               
            for gene in gene_off_list:
                cell_vs_gene[c_cell, gene] = min_lr_gene_exp #-10
                
            for cell in cell_neighborhood[c_cell]:            
                if cell in cell_of_interest:
                    continue
                if cell in active_spot:
                    continue
                    
                pattern_used[cell]=''
                for gene in gene_off_list:
                    cell_vs_gene[cell, gene] = min_lr_gene_exp #-10   
                        
    
            ##########################################
    
    
            #print('%d, %d, %d'%(a_cell, b_cell, c_cell))
            #pattern_used[a_cell] = ''
            #pattern_used[b_cell] = ''
            #pattern_used[c_cell] = ''
            
            pattern_used_list[pattern_type_index][a_cell] = ''
            pattern_used_list[pattern_type_index][b_cell] = ''
            pattern_used_list[pattern_type_index][c_cell] = ''
            '''
            if pattern_type_index == 0:
                pattern_used_0[a_cell] = ''
                pattern_used_0[b_cell] = ''
                pattern_used_0[c_cell] = ''
    
            if pattern_type_index == 1:
                pattern_used_1[a_cell] = ''
                pattern_used_1[b_cell] = ''
                pattern_used_1[c_cell] = ''
    
            if pattern_type_index == 2:
                pattern_used_2[a_cell] = ''
                pattern_used_2[b_cell] = ''
                #pattern_used_2[c_cell] = ''
            '''
    
            ##########################################
            '''
            for c in [i, cell_neighborhood[i][0], cell_neighborhood[cell_neighborhood[i][0]][0]]:
                if c in noise_cells:
                    for g in range (0, cell_vs_gene.shape[1]):
                        if cell_vs_gene[c][g] != 0: ## CHECK ##
                            cell_vs_gene[c, g] = cell_vs_gene[c, g] + gene_distribution_noise[c]
            '''
            for edge in edge_list:
                c1 = edge[0]
                c2 = edge[1]
                ligand_gene = edge[2]
                receptor_gene = edge[3]
                #########
                communication_score = cell_vs_gene[c1,ligand_gene] * cell_vs_gene[c2,receptor_gene] 
                #communication_score = max(communication_score, 0)
                if communication_score > 0:
                    lig_rec_dict_TP[c1][c2].append(ligand_dict_dataset[ligand_gene][receptor_gene])
                    P_class = P_class+1
                    p_dist.append(communication_score)
                else:
                    print('Error')
                #cells_ligand_vs_receptor[c1][c2].append([ligand_gene, receptor_gene, communication_score, ligand_dict_dataset[ligand_gene][receptor_gene]])              
                #########
    print('P_class %d'%P_class)                
    
    
    ############################
    '''
    
    0 - 8
    1 - 9
    2 - 10
    3 - 11
    
    4 - 12
    5 - 13
    6 - 14
    7 - 15
    
    '''
    

    ##############################
    cell_vs_gene_hold = copy.deepcopy(cell_vs_gene)
    #cell_vs_gene = copy.deepcopy(cell_vs_gene_hold)
    
    non_lr_cells = list(set(np.arange(cell_count)) -set(active_spot.keys()))
    np.random.shuffle(non_lr_cells)
    deactivate_count = (len(non_lr_cells)//3)*2
    #max_gene_exp = np.max(cell_vs_gene) 
    '''
    for cell in non_lr_cells: #[0:deactivate_count]:
        for lig_gene in ligand_dict_dataset:
            for rec_gene in ligand_dict_dataset[lig_gene]:
                if cell_vs_gene[cell,lig_gene]*cell_vs_gene[rec_gene,gene]<np.percentile(p_dist,50): # turn off so that only edges with high ccc exist in the input
                    cell_vs_gene[cell,lig_gene] = min_lr_gene_exp
                    cell_vs_gene[cell,rec_gene] = min_lr_gene_exp
    '''
    for cell in pattern_used.keys(): #non_lr_cells[0:deactivate_count]:
        for gene in ligand_gene_list:
            cell_vs_gene[cell,gene] = min_lr_gene_exp
        for gene in receptor_gene_list:
            cell_vs_gene[cell,gene] = min_lr_gene_exp
    
      
    #############################
    print("min value of cell_vs_gene before normalizing is %g"%np.min(cell_vs_gene))
    
    cell_vs_gene_notNormalized = copy.deepcopy(cell_vs_gene)
    # take quantile normalization.
    temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  #, axis=0
    adata_X = np.transpose(temp)  
    cell_vs_gene = adata_X
    print("min value of cell_vs_gene after normalizing is %g"%np.min(cell_vs_gene))
    
    #cell_vs_gene = copy.deepcopy(cell_vs_gene_notNormalized)
    
    
    cell_percentile = []
    for i in range (0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i])
        x = range(1, len(y)+1)
        kn = KneeLocator(x, y, curve='convex', direction='increasing')
        kn_value = y[kn.knee-1]    
        cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 96), np.percentile(y, 99) , kn_value])
    
    ###############
    
    # ready to go
    ################################################################################################
    # do the usual things
    ''''''
    cells_ligand_vs_receptor = []
    for i in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor.append([])
    
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[0]):
            cells_ligand_vs_receptor[i].append([])
            cells_ligand_vs_receptor[i][j] = []
     
    count = 0
    distribution_fp_score = []
    for i in range (0, cell_vs_gene.shape[0]): # ligand                 
        for j in range (0, cell_vs_gene.shape[0]): # receptor
            if dist_X[i,j] <= 0: #distance_matrix[i,j] > threshold_distance:
                continue
                  
            for gene in ligand_list:
                rec_list = list(ligand_dict_dataset[gene].keys())
                for gene_rec in rec_list:   
                    '''
                    if i in noise_cells:
                        cell_vs_gene[i][gene_index[gene]] = cell_vs_gene[i][gene_index[gene]] + gene_distribution_noise[i]
                    if j in noise_cells:
                        cell_vs_gene[j][gene_index[gene_rec]]  = cell_vs_gene[j][gene_index[gene_rec]]  + gene_distribution_noise[j]
                    '''  
                    
                    if cell_vs_gene[i][gene_index[gene]] > cell_percentile[i][2] and cell_vs_gene[j][gene_index[gene_rec]] > cell_percentile[j][2]:
                        communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]] #* dist_X[i,j]    
                        communication_score = max(communication_score, 0)
                        if communication_score>0:
                            cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, ligand_dict_dataset[gene][gene_rec]]) 
                            count = count + 1
                            if (i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and (lr_pair_type in lig_rec_dict_TP[i][j]))==False:
                                distribution_fp_score.append(communication_score*dist_X[i,j])
    
    
    print('total edges %d'%count)
    #################
    min_score = 1000
    max_score = -1000
    count = 0
    dist = []
    for i in range (0, len(lig_rec_dict_TP)):
        flag_debug = 0
        for j in range (0, len(lig_rec_dict_TP)):
            for l in range (0, len(lig_rec_dict_TP[i][j])):	
                flag_found = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    if lig_rec_dict_TP[i][j][l]==cells_ligand_vs_receptor[i][j][k][3]:
                        dist.append(cells_ligand_vs_receptor[i][j][k][2]*dist_X[i,j])
                        count = count + 1
                        flag_found=1
                        break
                #if flag_found==1:
    
    distribution_all_score = dist + distribution_fp_score               
    print('P_class=%d, found=%d, min %g, max %g, std %g, AVG %g, 80th of all %g, diff %g'%(P_class, count, np.min(dist), np.max(dist), np.std(dist), np.percentile(dist,50), np.percentile(distribution_all_score,80), np.percentile(distribution_all_score,80)-np.percentile(dist,50)))
    
    if np.percentile(dist,50)<=np.percentile(distribution_all_score,80):
        print('found')
        break
    
################
save_lig_rec_dict_TP = copy.deepcopy(lig_rec_dict_TP)
#lig_rec_dict_TP = copy.deepcopy(save_lig_rec_dict_TP)

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

lig_rec_dict_TP = 0            
lig_rec_dict_TP = lig_rec_dict_TP_temp
#################################################

ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((20))
fp_count = 0
for i in range (0, len(cells_ligand_vs_receptor)):
    for j in range (0, len(cells_ligand_vs_receptor)):
        if dist_X[i,j] > 0: #distance_matrix[i][j] <= threshold_distance: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                # if not
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):    
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    count_edge = count_edge + 1
                    count_local = count_local + 1
                    #print(count_edge)  
                    mean_ccc = cells_ligand_vs_receptor[i][j][k][2] 
                    #mean_ccc = .1 + (cells_ligand_vs_receptor[i][j][k][2]-min_score_global)/(max_score_global-min_score_global)*(1-0.1)   # cells_ligand_vs_receptor[i][j][k][2] #cells_ligand_vs_receptor[i][j][k][2]  #*dist_X[i,j]
                    lr_pair_type = cells_ligand_vs_receptor[i][j][k][3]
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and (lr_pair_type in lig_rec_dict_TP[i][j]):               
                        row_col.append([i,j])
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        edge_weight.append([dist_X[i,j], mean_ccc, cells_ligand_vs_receptor[i][j][k][3] ])
                        lig_rec.append(cells_ligand_vs_receptor[i][j][k][3])
                    else: # mean_ccc >= np.percentile(distribution_fp_score,50): #if fp_count<=100000:
                        row_col.append([i,j])
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        edge_weight.append([dist_X[i,j], mean_ccc, cells_ligand_vs_receptor[i][j][k][3] ])
                        lig_rec.append(cells_ligand_vs_receptor[i][j][k][3])    
                        fp_count = fp_count + 1
                

                
                if max_local < count_local:
                    max_local = count_local
            '''       
            else: #elif i in pattern_used and j in pattern_used:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', '']),
        	'''
                
            ''' '''
            #local_list[count_local] = local_list[count_local] + 1


		
print('len row col %d'%len(row_col))
print('max local %d'%max_local) 
#print('random_activation %d'%len(random_activation_index))
print('ligand_cells %d'%len(ligand_cells))
print('P_class %d'%P_class) 




#options = options+ '_' + 'wFeature'

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'cellvsgene', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'wb') as fp:
    pickle.dump(cell_vs_gene_notNormalized, fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options, 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec], fp)

random_activation = []
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([lr_database, lig_rec_dict_TP, random_activation], fp)

ccc_region = active_spot
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'wb') as fp:
    pickle.dump([temp_x, temp_y, ccc_region], fp)

cell_vs_lrgene = cell_vs_gene[:,0:lr_gene_count]
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'cellvslrgene', 'wb') as fp:
    pickle.dump(cell_vs_lrgene, fp)


'''
for index in range (0, len(row_col)):
    if lig_rec[index] == 1:
        lig_rec[index] = 5
        
    elif lig_rec[index] == 5:
        lig_rec[index] = 1
        
    elif lig_rec[index] == 3:
        lig_rec[index] = 6
        
    elif lig_rec[index] == 6:
        lig_rec[index] = 3
        
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options+'_swappedLRid', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec], fp)        
        
for i in lig_rec_dict_TP:
    for j in lig_rec_dict_TP[i]:
        for k in range (0, len(lig_rec_dict_TP[i][j])):
            if lig_rec_dict_TP[i][j][k] == 1:
                lig_rec_dict_TP[i][j][k] = 5
            elif lig_rec_dict_TP[i][j][k] == 5:
                lig_rec_dict_TP[i][j][k] = 1
            elif lig_rec_dict_TP[i][j][k] == 3:
                lig_rec_dict_TP[i][j][k] = 6
            elif lig_rec_dict_TP[i][j][k] == 6:
                lig_rec_dict_TP[i][j][k] = 3
                
a = lr_database[1]
lr_database[1] = lr_database[5]
lr_database[5] = a

a = lr_database[3]
lr_database[3] = lr_database[6]
lr_database[6] = a 



with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options+'_swappedLRid', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([lr_database, lig_rec_dict_TP, random_activation], fp)
                        
'''

'''
# to preprocess it for COMMOT and Niches
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'rb') as fp:
    temp_x, temp_y, ccc_region  = pickle.load(fp)

data_list_pd = pd.DataFrame(temp_x)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_x.csv', index=False, header=False)
data_list_pd = pd.DataFrame(temp_y)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_y.csv', index=False, header=False)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'rb') as fp:
    cell_vs_gene = pickle.load(fp)

data_list=defaultdict(list)
for i in range (0, cell_vs_gene.shape[0]):
    #max_value=np.max(cell_vs_gene[i])
    #min_value=np.min(cell_vs_gene[i])
    for j in range (0, cell_vs_gene.shape[1]):
        data_list['a-'+str(i)].append(cell_vs_gene[i][j]) #(cell_vs_gene[i][j]-min_value)/(max_value-min_value)
        
        
data_list_pd = pd.DataFrame(data_list)    
gene_name = []
for i in range (0, cell_vs_gene.shape[1]):
    gene_name.append('g'+str(i))
    
data_list_pd[' ']=gene_name   
data_list_pd = data_list_pd.set_index(' ')    
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_g.csv')

data_list=dict()
data_list['ligand']=[]
data_list['receptor']=[]
for i in range (0, len(lr_database)):
    data_list['ligand'].append('g'+str(lr_database[i][0]))
    data_list['receptor'].append('g'+str(lr_database[i][1]))
    
data_list_pd = pd.DataFrame(data_list)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_lr_type6_g.csv', index=False)
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
#options = 'dt-'+datatype+'_lrc'+str(25)+'_cp'+str(cell_percent)+'_np'+str(neighbor_percent)+'_lrp'+str(lr_percent)+'_'+receptor_connections

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

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
    row_col, edge_weight, lig_rec  = pickle.load(fp)  #, lr_database, lig_rec_dict_TP, random_activation
    

	
max_tp_distance = 0
datapoint_size = temp_x.shape[0]              
total_type = np.zeros((len(lr_database)))
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i==j: 
            continue
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
            for k in range (0, len(lig_rec_dict_TP[i][j])):
                total_type[lig_rec_dict_TP[i][j][k]] = total_type[lig_rec_dict_TP[i][j][k]] + 1
                if max_tp_distance<distance_matrix[i,j]:
                    max_tp_distance = distance_matrix[i,j]
count = 0
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    if i!=j:
	    count = count +1     
		
positive_class = np.sum(total_type)
negative_class = count - positive_class           
############# draw the points which are participating in positive classes  ######################
ccc_index_dict = dict()     
for i in lig_rec_dict_TP:
    ccc_index_dict[i] = ''
    for j in lig_rec_dict_TP[i]:
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
###########
'''
plt.hist(distribution, color = 'blue', bins = int(len(distribution)/5))
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'distribution_type6_f_input.svg', dpi=400)
plt.clf()
'''
'''
###########
# split it into two set of edges
    ###########
    dict_cell_edge = defaultdict(list) # incoming edges
    dict_cell_neighbors = defaultdict(list) # incoming edges
    for i in range(0, len(row_col)):
        dict_cell_edge[row_col[i][1]].append(i) # index
        dict_cell_neighbors[row_col[i][1]].append(row_col[i][0])

    for i in range (0, datapoint_size):
        neighbor_list = dict_cell_neighbors[i]
        neighbor_list = list(set(neighbor_list))
        dict_cell_neighbors[i] = neighbor_list

    set1_nodes = []
    set1_edges_index = []
    node_limit_set1 = datapoint_size//2
    set1_direct_edges = []
    print('set 1 has nodes upto: %d'%node_limit_set1)
    for i in range (0, node_limit_set1):
        set1_nodes.append(i)
        # add it's edges - first hop
        for edge_index in dict_cell_edge[i]:
            set1_edges_index.append(edge_index) # has both row_col and edge_weight
            set1_direct_edges.append(edge_index)
        # add it's neighbor's edges - second hop
        for neighbor in dict_cell_neighbors[i]:
            if i == neighbor:
                continue
            for edge_index in dict_cell_edge[neighbor]:
                set1_edges_index.append(edge_index) # has both row_col and edge_weight

    set1_edges_index = list(set(set1_edges_index))
    print('amount of edges in set 1 is: %d'%len(set1_edges_index))

    set2_nodes = []
    set2_edges_index = []
    set2_direct_edges = []
    print('set 2 has nodes upto: %d'%datapoint_size)
    for i in range (node_limit_set1, datapoint_size):
        set2_nodes.append(i)
        # add it's edges - first hop
        for edge_index in dict_cell_edge[i]:
            set2_edges_index.append(edge_index) # has both row_col and edge_weight
            set2_direct_edges.append(edge_index)
        # add it's neighbor's edges - second hop
        for neighbor in dict_cell_neighbors[i]:
            if i == neighbor:
                continue
            for edge_index in dict_cell_edge[neighbor]:
                set2_edges_index.append(edge_index) # has both row_col and edge_weight

    set2_edges_index = list(set(set2_edges_index))
    print('amount of edges in set 1 is: %d'%len(set2_edges_index))

    set1_edges = []
    for i in range (0, len(set1_direct_edges)): #len(set1_edges_index)
        set1_edges.append([row_col[i], edge_weight[i]])

    set2_edges = []
    for i in range (0, len(set2_direct_edges)): #set2_edges_index
        set2_edges.append([row_col[i], edge_weight[i]])
'''
##################################################
'''
for i in range (0, datapoint_size):  
    for j in range (0, datapoint_size):	
        if i in ccc_index_dict and j in ccc_index_dict:
            for k in range (0, len(lig_rec_dict[i][j])):
                if j not in lig_rec_dict_TP[i]:
                    lig_rec_dict_TP[i][j] = []
                lig_rec_dict_TP[i][j].append(lig_rec_dict[i][j][k])
        
ccc_index_dict = dict()  
P_class = 0
for i in lig_rec_dict_TP:
    ccc_index_dict[i] = ''
    for j in lig_rec_dict_TP[i]:
        ccc_index_dict[j] = ''  
        P_class = P_class + len(lig_rec_dict_TP[i][j])
'''        
############

plot_dict = defaultdict(list)
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
    #positive_class = 0  
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue

            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #positive_class = positive_class + 1                     
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                        #else:
                        #    confusion_matrix[0][1] = confusion_matrix[0][1] + 1                 
                    else:
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                        #else:
                        #    confusion_matrix[1][1] = confusion_matrix[1][1] + 1      

    print('%d, %g, %g'%(percentage_value, (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    FPR_value = (confusion_matrix[1][0]/negative_class)#*100
    TPR_value = (confusion_matrix[0][0]/positive_class)#*100
    plot_dict['FPR'].append(FPR_value)
    plot_dict['TPR'].append(TPR_value)
    plot_dict['Type'].append('naive_model')

###########################################   
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 10
csv_record_dict = defaultdict(list)
for run_time in range (0,total_runs):
    run = run_time
    #if run in [1, 2, 4, 7, 8]:
    #    continue

    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_6_path_knn10_f_3d_'+filename[run]+'_attention_l1.npy'
    X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_6_path_knn10_f_tanh_3d_dropout_'+filename[run]+'_attention_l1.npy' #split_
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_tanh_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3 #_swappedLRid
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_relu_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_tanh_3dim_dropout_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 
    # [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized, X_attention_score_unnormalized_l1, X_attention_score_normalized]
    l=2 #2 ## 
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
    plt.hist(distribution, color = 'blue', bins = int(len(distribution)/5))
    save_path = '/cluster/home/t116508uhn/64630/'
    #plt.savefig(save_path+'distribution_type6_f_3d_tanh_'+filename[run]+'.svg', dpi=400)
    #plt.savefig(save_path+'distribution_e_3d_tanh_swappedLRid_'+filename[run]+'.svg', dpi=400)
    #plt.savefig(save_path+'distribution_e_3d_relu_'+filename[run]+'.svg', dpi=400)
    #plt.savefig(save_path+'distribution_e_3d_gatconv_'+filename[run]+'.svg', dpi=400)
    #plt.savefig(save_path+'distribution_type6_f_3d_tanh_'+filename[run]+'.svg', dpi=400)
    #plt.savefig(save_path+'distribution_type6_f_3d_'+filename[run]+'.svg', dpi=400)
    plt.clf()
    
    percentage_value = 100
    while percentage_value > 0:
        #distribution_partial = []
        percentage_value = percentage_value - 10
    #for percentage_value in [79, 85, 90, 93, 95, 97]:
        existing_lig_rec_dict = []
        datapoint_size = temp_x.shape[0]
        count = 0
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
                        count = count + 1
                        #distribution_partial.append(attention_scores[i][j][k])

        #############
        #positive_class = 0  
        #negative_class = 0
        confusion_matrix = np.zeros((2,2))
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):

                if i==j: 
                    continue

                if len(existing_lig_rec_dict[i][j])>0:
                    for k in existing_lig_rec_dict[i][j]:   
                        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                            #positive_class = positive_class + 1                     
                            confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                            #else:
                            #    confusion_matrix[0][1] = confusion_matrix[0][1] + 1                 
                        else:
                            confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                            #else:
                            #    confusion_matrix[1][1] = confusion_matrix[1][1] + 1      

        print('%d, %g, %g'%(percentage_value, (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))
        FPR_value = (confusion_matrix[1][0]/negative_class)#*100
        TPR_value = (confusion_matrix[0][0]/positive_class)#*100
        plot_dict['FPR'].append(FPR_value)
        plot_dict['TPR'].append(TPR_value)
        plot_dict['Type'].append('run_'+str(run+1))
        
        #plt.hist(distribution_partial, color = 'blue', bins = int(len(distribution_partial)/5))
        #save_path = '/cluster/home/t116508uhn/64630/'
        #plt.savefig(save_path+'distribution_e_3d_relu_partial_'+filename[run]+'_'+str(percentage_value)+'.svg', dpi=400)
        #plt.savefig(save_path+'distribution_e_3d_gatconv_'+filename[run]+'.svg', dpi=400)
        #plt.savefig(save_path+'distribution_e_3d_tanh_'+filename[run]+'.svg', dpi=400)
        #plt.clf()


data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:Q',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/64630/'
#chart.save(save_path+'plot_type4_e_leakyrelu.html')
chart.save(save_path+'plot_type4_e_3d_tanh_dropout_layer2attention.html')
#chart.save(save_path+'plot_e_gatconv.html')
#chart.save(save_path+'plot_type6_f_3d_tanh_dropout_layer2attention.html')
#chart.save(save_path+'plot_e_relu.html')
chart.save(save_path+'plot_type4_e_3d_1layer.html')


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
                nt.add_edge(int(i), int(j), label=title_str) #, value=np.float64(attention_scores[i][j][k])) #,width=, arrowsize=int(20),  arrowstyle='fancy'

nt.show('mygraph.html')

#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html
#################################################################################
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

#df_pair_vs_cells = pd.read_csv('/cluster/home/t116508uhn/niches_output_PDAC_pair_vs_cells.csv')
df_pair_vs_cells = pd.read_csv('/cluster/home/t116508uhn/niches_output_pair_vs_cells_type6_f.csv')
#df_cells_vs_cluster = pd.read_csv('/cluster/home/t116508uhn/niches_output_cluster_vs_cells.csv')
distribution = []
for col in range (1, len(df_pair_vs_cells.columns)):
    col_name = df_pair_vs_cells.columns[col]
    l_c = df_pair_vs_cells.columns[col].split("—")[0]
    r_c = df_pair_vs_cells.columns[col].split("—")[1]
    l_c = l_c.split('.')[1]
    r_c = r_c.split('.')[1]
    i = int(l_c)
    j = int(r_c)
    
    for index in range (0, len(df_pair_vs_cells.index)):
        lig_rec_dict[i][j].append(df_pair_vs_cells.index[index])
        attention_scores[i][j].append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])
        distribution.append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])

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
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue
            if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i]:
                for k in range (0, len(lig_rec_dict_TP[i][j])):
                    if k in existing_lig_rec_dict[i][j]:
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[0][1] = confusion_matrix[0][1] + 1 

            '''
            if len(lig_rec_dict[i][j])>0:
                for k in lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        
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
              '''
    print('%d, %g'%(percentage_value, (confusion_matrix[0][0]/positive_class)*100))    
    
import altair as alt
from vega_datasets import data


######################################
FPR = [0, 7.72522, 18.3623, 30.5919, 42.7621, 54.7778, 65.9853, 76.135, 88.0556, 95.6145, 100]
TPR_naive = [0, 0, 0, 0, 0, 0, 0, 1.82403, 12.6609, 38.5193, 100]
TPR_us= [50, 50, 50.4292, 56.7597, 62.7682, 70.6009, 75, 82.8326, 91.4163, 95, 100]

plot_dict = defaultdict(list)

for i in range (0, len(FPR)):
    plot_dict['FPR'].append(FPR[i])
    plot_dict['TPR'].append(TPR_naive[i])
    plot_dict['Type'].append('naive_model')
    
for i in range (0, len(FPR)):
    plot_dict['FPR'].append(FPR[i])
    plot_dict['TPR'].append(TPR_us[i])
    plot_dict['Type'].append('our_model')
    
    
data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:T',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'plot_e_tanh.html')



