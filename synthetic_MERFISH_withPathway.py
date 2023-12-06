import os
#import glob
import pandas as pd
#import shutil 
import copy
import csv
import numpy as np
import sys
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

import altairThemes
import altair as alt

alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")

 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()

threshold_distance = 4 #2 = path equally spaced
k_nn = 10 # #5 = h
distance_measure = 'threshold_dist' #'knn'  # <-----------
datatype = 'path_uniform_distribution' #'path_equally_spaced' #

'''
distance_measure = 'knn'  #'threshold_dist' # <-----------
datatype = 'pattern_high_density_grid' #'pattern_equally_spaced' #'mixture_of_distribution' #'equally_spaced' #'high_density_grid' 'uniform_normal' # <-----------'dt-pattern_high_density_grid_lrc1_cp20_lrp1_randp0_all_same_midrange_overlap'
'''
cell_percent = 100 # choose at random N% ligand cells

lr_gene_count = 44 #24 #8 #100 #20 #100 #20 #50 # and 25 pairs
rec_start = lr_gene_count//2 # 

ligand_gene_list = []
for i in range (0, rec_start):
    ligand_gene_list.append(i)

receptor_gene_list = []
for i in range (rec_start, lr_gene_count):
    receptor_gene_list.append(i)

# ligand_gene_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# receptor_gene_list = [22,23,24, 25, 26, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42, 43]


non_lr_genes = 350 - lr_gene_count

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

###############################################
lr_database = []
for i in range (0, 12): #len(ligand_gene_list)):
    lr_database.append([ligand_gene_list[i],receptor_gene_list[i]])

ligand_dict_dataset = defaultdict(dict)
for i in range (0, len(lr_database)):
    ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i
ligand_list = list(ligand_dict_dataset.keys())  


# just print the lr_database
for i in range (0, len(lr_database)):
    print('%d: %d - %d'%(i, lr_database[i][0], lr_database[i][1]))
'''
0: 0 - 22
1: 1 - 23
2: 2 - 24
3: 3 - 25
4: 4 - 26
5: 5 - 27
6: 6 - 28
7: 7 - 29
8: 8 - 30
9: 9 - 31
10: 10 - 32
11: 11 - 33
12: 12 - 34
13: 13 - 35
14: 14 - 36
15: 15 - 37
16: 16 - 38
17: 17 - 39
18: 18 - 40
19: 19 - 41
20: 20 - 42
21: 21 - 43
'''
#pattern_list = [[[0, 1],[2, 3]], [[4, 5], [6, 7]]]

max_lr_pair_id = 80 #len(lr_database)//2
connection_count_max = 2 # for each pair of cells
pattern_list = []
i = 0
stop_flag = 0
while i < (len(lr_database)-connection_count_max*2):
    pattern = []
    j = i
    connection_count = 0
    while connection_count < connection_count_max:
        pattern.append([j, j+1])
        j = j + 2
        connection_count = connection_count + 1
        if j == max_lr_pair_id or j+1 == max_lr_pair_id:
            stop_flag = 1
            break
            
    i = j
    if stop_flag==1:
        continue
    pattern_list.append(pattern)

pattern_list = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
'''
In [8]: pattern_list
Out[8]: 
[[[0, 1], [2, 3]],
 [[4, 5], [6, 7]],
 [[8, 9], [10, 11]],
 [[12, 13], [14, 15]],
 [[16, 17], [18, 19]]]
'''


# just print the lr_database
TP_LR_genes = []
for i in range (0, len(lr_database)):
    print('%d: %d - %d'%(i, lr_database[i][0], lr_database[i][1]))
    TP_LR_genes.append(lr_database[i][0])
    TP_LR_genes.append(lr_database[i][1])
    
################# Now create some arbitrary pairs that will be false positives #########

for i in range (12, len(ligand_gene_list)):
#    lr_database.append([ligand_gene_list[i],receptor_gene_list[i]])
#    lr_database.append([ligand_gene_list[i],receptor_gene_list[i+1]])
#    lr_database.append([ligand_gene_list[i],receptor_gene_list[i+2]])
    for j in range (12, len(receptor_gene_list)):
        lr_database.append([ligand_gene_list[i],receptor_gene_list[j]])

ligand_dict_dataset = defaultdict(dict)
for i in range (0, len(lr_database)):
    ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i
    
ligand_list = list(ligand_dict_dataset.keys())  
''''''


########################################################################################

noise_add = 0  #2 #1
noise_percent = 0 #30
random_active_percent = 0
active_type = 'random_overlap' #'highrange_overlap' #


def get_data(datatype):
    if datatype == 'path_equally_spaced':
        x_max = 50 #50 
        x_min = 0
        y_max = 60 #20 #30 
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
    
    elif datatype == 'path_uniform_distribution':
	
        datapoint_size = 5000
        x_max = 150 #500
        x_min = 0
        y_max = 150 #300
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
        '''
        for i in range (0, temp_x.shape[0]):
            if i not in discard_points:
                for j in range (i+1, temp_x.shape[0]):
                    if j not in discard_points:
                        if euclidean_distances(np.array([[temp_x[i],temp_y[i]]]), np.array([[temp_x[j],temp_y[j]]]))[0][0] < 1 :
                            print('i: %d and j: %d'%(i,j))
                            discard_points[j]=''
        '''
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
plt.gca().set_aspect(1)	
plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'synthetic_spatial_plot_'+datatype+'.svg', dpi=400)
plt.clf()


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
    cell_neighborhood_temp = sorted(cell_neighborhood_temp, key = lambda x: x[1], reverse=True) # sort based on distance
    
    cell_neighborhood[cell] = [] # to record the neighbor cells in that order
    for items in cell_neighborhood_temp:
        cell_neighborhood[cell].append(items[0])
    #np.random.shuffle(cell_neighborhood[cell]) 
####################################################################################            
# take lr_gene_count normal distributions where each distribution has len(temp_x) datapoints.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
i_am_whose = []
for i in range (0, datapoint_size):
    i_am_whose.append([])

for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i in cell_neighborhood[j] and j not in cell_neighborhood[i]:
            i_am_whose[i].append(j)
            
    

max_neighbor = 0
for i in range (0, len(cell_neighborhood)):
    if len(cell_neighborhood[i])>max_neighbor:
        max_neighbor = len(cell_neighborhood[i])
print('max neighborhood: %d'%max_neighbor)


cell_count = len(temp_x)
gene_distribution_active = np.zeros((lr_gene_count + non_lr_genes, cell_count))
gene_distribution_inactive = np.zeros((lr_gene_count + non_lr_genes, cell_count))
#gene_distribution_inactive_lrgenes = np.zeros((lr_gene_count + non_lr_genes, cell_count))
gene_distribution_noise = np.zeros((lr_gene_count + non_lr_genes, cell_count))


################
start_loc = 20
rec_gene = lr_gene_count//2
for i in range (0, 12):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x))   #loc=start_loc+(i%15) from loc=start_loc+(i%5) -- gave more variations so more FP
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i,:] =  gene_exp_list
    #print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x))
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[rec_gene ,:] =  gene_exp_list
    #print('%d: inactive: %g to %g'%(rec_gene, np.min(gene_distribution_inactive[rec_gene,:]),np.max(gene_distribution_inactive[rec_gene,:]) ))
    rec_gene = rec_gene + 1 
    # np.min(gene_distribution_inactive[i,:])-3, scale=.5


start_loc = 20
for i in range (12, lr_gene_count//2):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x))   #loc=start_loc+(i%15) from loc=start_loc+(i%5) -- gave more variations so more FP
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i,:] =  gene_exp_list

    print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x))
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[rec_gene ,:] =  gene_exp_list
    #print('%d: inactive: %g to %g'%(rec_gene, np.min(gene_distribution_inactive[rec_gene,:]),np.max(gene_distribution_inactive[rec_gene,:]) ))
    rec_gene = rec_gene + 1
    # np.min(gene_distribution_inactive[i,:])-3, scale=.5



'''
cell_dummy = np.arrange(len(temp_x))
np.random.shuffle(cell_dummy) 
start_loc = 20
rec_gene_save = rec_gene
for i in range (12, lr_gene_count//2):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x)//2)   #loc=start_loc+(i%15) from loc=start_loc+(i%5) -- gave more variations so more FP
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i,cell_dummy[0:len(cell_dummy)//2]] =  gene_exp_list


	
    #print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x)//2)
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[rec_gene ,cell_dummy[0:len(cell_dummy)//2]] =  gene_exp_list
    #print('%d: inactive: %g to %g'%(rec_gene, np.min(gene_distribution_inactive[rec_gene,:]),np.max(gene_distribution_inactive[rec_gene,:]) ))
    rec_gene = rec_gene + 1
    # np.min(gene_distribution_inactive[i,:])-3, scale=.5

################
'''
'''
start_loc = 20
rec_gene = rec_gene_save 
for i in range (12, lr_gene_count//2): ##):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x)//2)
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i, cell_dummy[len(cell_dummy)//2:]] =  gene_exp_list
    print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))
    
    ###############

    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=2,size=len(temp_x)//2)
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[rec_gene, cell_dummy[len(cell_dummy)//2:]] =  gene_exp_list
    print('%d: inactive: %g to %g'%(rec_gene, np.min(gene_distribution_inactive[rec_gene,:]),np.max(gene_distribution_inactive[rec_gene,:]) ))
    ###################

    rec_gene = rec_gene + 1 

###################################################
'''


start_loc = 15
for i in range (rec_gene, lr_gene_count + non_lr_genes):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=3,size=len(temp_x))
    np.random.shuffle(gene_exp_list) 
    gene_distribution_inactive[i,:] =  gene_exp_list
    #print('%d: inactive: %g to %g'%(i, np.min(gene_distribution_inactive[i,:]),np.max(gene_distribution_inactive[i,:]) ))

    
    
#################
start_loc = np.max(gene_distribution_inactive)+30
rec_gene = lr_gene_count//2
scale_active_distribution = 1 #0.01
for i in range (0, 4):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=len(temp_x)) #
    np.random.shuffle(gene_exp_list) 
    gene_distribution_active[i,:] =  gene_exp_list
    #print('%d: active: %g to %g'%(i, np.min(gene_distribution_active[i,:]),np.max(gene_distribution_active[i,:]) ))
    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=len(temp_x)) #
    np.random.shuffle(gene_exp_list) 
    gene_distribution_active[rec_gene ,:] =  gene_exp_list
    #print('%d: active: %g to %g'%(rec_gene, np.min(gene_distribution_active[rec_gene,:]),np.max(gene_distribution_active[rec_gene,:]) ))
    rec_gene = rec_gene + 1 

#start_loc = 30 #np.max(gene_distribution_inactive)+2
for i in range (4, lr_gene_count//2):
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=len(temp_x)) #
    np.random.shuffle(gene_exp_list) 
    gene_distribution_active[i,:] =  gene_exp_list
    #print('%d: active: %g to %g'%(i, np.min(gene_distribution_active[i,:]),np.max(gene_distribution_active[i,:]) ))
    
    gene_exp_list = np.random.normal(loc=start_loc+(i%5),scale=scale_active_distribution,size=len(temp_x)) #
    np.random.shuffle(gene_exp_list) 
    gene_distribution_active[rec_gene ,:] =  gene_exp_list
    #print('%d: active: %g to %g'%(rec_gene, np.min(gene_distribution_active[rec_gene,:]),np.max(gene_distribution_active[rec_gene,:]) ))
    rec_gene = rec_gene + 1 
   
#################################################
min_lr_gene_count = np.min(gene_distribution_inactive)

print('min_lr_gene_count %d'%min_lr_gene_count)
#min_lr_gene_count = 0
    
#########################      	
cell_vs_gene = np.zeros((cell_count,lr_gene_count + non_lr_genes))
# initially all are in inactive state
for i in range (0, lr_gene_count + non_lr_genes):
    cell_vs_gene[:,i] = gene_distribution_inactive[i,:]
###############################################################

# record true positive connections    
lig_rec_dict_TP = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size): 
    lig_rec_dict_TP.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict_TP[i].append([])   
        lig_rec_dict_TP[i][j] = []
	
P_class = 0

active_spot_in_pattern = []
neighbour_of_actives_in_pattern = []
for i in range (0, len(pattern_list)):
    active_spot_in_pattern.append(dict())
    neighbour_of_actives_in_pattern.append(dict())

active_spot = dict()
neighbour_of_actives = dict()

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
flag_stop = 0
pattern_count = len(pattern_list)
for pattern_type in range (0, 3): #8): #pattern_count):	
    discard_cells = list(active_spot.keys()) # + list(neighbour_of_actives.keys())  
    ligand_cells = list(set(np.arange(cell_count)) - set(discard_cells))
    max_ligand_count = 250 #100 #cell_count//(pattern_count*6) # 10.  1/N th of the all cells are following this pattern, where, N = total patterns
    np.random.shuffle(ligand_cells)
    print("pattern_type_index %d, ligand_cell count %d"%(pattern_type, max_ligand_count ))
    #print(ligand_cells[0:10])
    
    set_ligand_cells = []
    for i in ligand_cells:
        set_ligand_cells.append([temp_x[i], temp_y[i]]) 
    
    k= -1
    for i in ligand_cells:
        # choose which L-R are working for this ligand i        
        if k > max_ligand_count:
            break        
        a_cell = i
        temp_neighborhood = []
        for neighbor_cell in cell_neighborhood[a_cell]:
            if neighbor_cell != a_cell:
                temp_neighborhood.append(neighbor_cell)
       
        if (len(temp_neighborhood)<1):
            continue    
            
        b_cell = temp_neighborhood[len(temp_neighborhood)-1]  # take the last one to make the pattern complex
        temp_neighborhood = []
	    

        for neighbor_cell in cell_neighborhood[b_cell]:
            if neighbor_cell != a_cell and neighbor_cell != b_cell:
                temp_neighborhood.append(neighbor_cell)
                
        if len(temp_neighborhood)<1:
            continue

        c_cell = temp_neighborhood[len(temp_neighborhood)-1]  # take the last one to make the pattern complex
                
        
        
        #if a_cell in neighbour_of_actives or b_cell in neighbour_of_actives or c_cell in neighbour_of_actives:
        #    continue
        
        if a_cell in active_spot or b_cell in active_spot or c_cell in active_spot:
            continue


        if a_cell in neighbour_of_actives_in_pattern[pattern_type] or b_cell in neighbour_of_actives_in_pattern[pattern_type] or c_cell in neighbour_of_actives_in_pattern[pattern_type]:
            continue
            
	    
   
        #if a_cell in active_spot_in_pattern[pattern_type] or b_cell in active_spot_in_pattern[pattern_type] or c_cell in active_spot_in_pattern[pattern_type]: # or  cell_neighborhood[cell_neighborhood[cell_neighborhood[i][0]][0]][0] in neighbour_of_actives:
        #    continue        
            
        gene_group = pattern_list[pattern_type]    
      

        k = k + 1 
        ##########################################  
        a_cell_active_genes = []
        b_cell_active_genes = []
        c_cell_active_genes = []
        edge_list = []
        ###########################################
        for gene_pair in gene_group:
            a = gene_pair[0]
            b = gene_pair[1]
        
            lr_i = a
            ligand_gene = lr_database[lr_i][0]
            receptor_gene = lr_database[lr_i][1]
            cell_id = a_cell
            cell_vs_gene[cell_id, ligand_gene] = gene_distribution_active[ligand_gene, cell_id]
            a_cell_active_genes.append(ligand_gene)
            
            cell_id = b_cell
            cell_vs_gene[cell_id, receptor_gene] = gene_distribution_active[receptor_gene, cell_id]
            b_cell_active_genes.append(receptor_gene)
            edge_list.append([a_cell, b_cell, ligand_gene, receptor_gene])
    
            #########################################
            
            lr_i = b
            ligand_gene = lr_database[lr_i][0]
            receptor_gene = lr_database[lr_i][1]
            cell_id = b_cell
            cell_vs_gene[cell_id, ligand_gene] = gene_distribution_active[ligand_gene, cell_id]
            b_cell_active_genes.append(ligand_gene)
            
            cell_id = c_cell
            cell_vs_gene[cell_id, receptor_gene] = gene_distribution_active[receptor_gene, cell_id]
            edge_list.append([b_cell, c_cell, ligand_gene, receptor_gene])
            c_cell_active_genes.append(receptor_gene)

        #################


        ligand_receptor_genes = ligand_gene_list + receptor_gene_list
        for gene in ligand_receptor_genes:
            if gene not in a_cell_active_genes:
                cell_vs_gene[a_cell, gene] = min_lr_gene_count #-10
                
        for gene in ligand_receptor_genes:
            if gene not in b_cell_active_genes:
                cell_vs_gene[b_cell, gene] = min_lr_gene_count #-10
                
        for gene in ligand_receptor_genes:
            if gene not in c_cell_active_genes:
                cell_vs_gene[c_cell, gene] = min_lr_gene_count #-10



        ##########################################


        #print('%d, %d, %d'%(a_cell, b_cell, c_cell))
        gene_off_list = a_cell_active_genes + b_cell_active_genes + c_cell_active_genes # all the ligand, receptor genes involve in this pattern
        gene_off_list = list(set(gene_off_list )) # to remove duplicate entries

        ################################
	    # extend this list by adding the ligand / receptor who are involved with gene_off_list
        '''
        additional_gene = []
        for gene in ligand_gene_list:
            # if there is any ligand gene who has a receptor gene in gene_off_list, the add that ligand gene to the list as well
            for receptor_gene in list(ligand_dict_dataset[gene].keys()):
                if receptor_gene in gene_off_list:
                    additional_gene.append(gene)
                    break
            
        for gene in gene_off_list:
            if gene in ligand_gene_list:
            # all receptor genes of this ligand gene should be included to the list as well
                for receptor_gene in list(ligand_dict_dataset[gene].keys()):
                    additional_gene.append(receptor_gene)
                    
        gene_off_list = gene_off_list + additional_gene
        gene_off_list = list(set(gene_off_list )) # to remove duplicate entries    
        '''
        ################################

        active_spot[a_cell] = ''
        active_spot[b_cell] = ''
        active_spot[c_cell] = ''
        turn_off_cell_list = list(set(cell_neighborhood[a_cell] + i_am_whose[a_cell]))
        for cell in turn_off_cell_list: #cell_neighborhood[a_cell]:
            
            if cell in [a_cell, b_cell, c_cell]:
                continue
            if cell in active_spot:
                continue
                
            neighbour_of_actives[cell]=''
            neighbour_of_actives_in_pattern[pattern_type][cell] = ''
            for gene in gene_off_list: #[0, 1, 2, 3,  8, 9, 10, 11]:
                cell_vs_gene[cell, gene] = min_lr_gene_count #-10   

        turn_off_cell_list = list(set(cell_neighborhood[b_cell] + i_am_whose[b_cell]))
        for cell in turn_off_cell_list:
            
            if cell in [a_cell, b_cell, c_cell]:
                continue
            if cell in active_spot:
                continue
                
            neighbour_of_actives[cell]=''
            neighbour_of_actives_in_pattern[pattern_type][cell] = ''
            for gene in gene_off_list: #[0, 1, 2, 3,  8, 9, 10, 11]:
                cell_vs_gene[cell, gene] = min_lr_gene_count #-10

        turn_off_cell_list = list(set(cell_neighborhood[c_cell] + i_am_whose[c_cell]))
        for cell in turn_off_cell_list:
            
            if cell in [a_cell, b_cell, c_cell]:
                continue
            if cell in active_spot:
                continue
                
            neighbour_of_actives[cell]=''
            neighbour_of_actives_in_pattern[pattern_type][cell] = ''
            for gene in gene_off_list: #[0, 1, 2, 3,  8, 9, 10, 11]:
                cell_vs_gene[cell, gene] = min_lr_gene_count #-10
            
        active_spot_in_pattern[pattern_type][a_cell] = ''
        active_spot_in_pattern[pattern_type][b_cell] = ''
        active_spot_in_pattern[pattern_type][c_cell] = ''

        ##########################################

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
            else:
                print('zero value found %g'%communication_score )
                flag_stop = 1
                break
            #cells_ligand_vs_receptor[c1][c2].append([ligand_gene, receptor_gene, communication_score, ligand_dict_dataset[ligand_gene][receptor_gene]])              
            #########
        if flag_stop == 1:
            break
    print('pattern %d is formed %d times'%(pattern_type, k))

print('P_class %d'%P_class)                

cell_vs_gene_org = copy.deepcopy(cell_vs_gene)

if noise_percent > 0:
    cell_count = cell_vs_gene.shape[0]
    if noise_add == 1:
        noise_percent = 30
        noise_cells = list(np.random.randint(0, cell_count, size=(cell_count*noise_percent)//100)) #“discrete uniform” distribution #ccc_region #
        gene_distribution_noise = np.random.normal(loc=0, scale=1, size = (len(noise_cells), cell_vs_gene.shape[1]))
        np.random.shuffle(gene_distribution_noise)	
        print('noise: %g to %g'%(np.min(gene_distribution_noise),np.max(gene_distribution_noise) ))
    elif noise_add == 2:
        noise_percent = 30
        '''    
        discard_cells = list(active_spot.keys()) 
        noise_cells = list(set(np.arange(cell_count)) - set(discard_cells))
        np.random.shuffle(noise_cells)	
        noise_cells = noise_cells[0:(cell_count*noise_percent)//100]
        '''    
        noise_cells = list(np.random.randint(0, cell_count, size=(cell_count*noise_percent)//100)) #“discrete uniform” distribution #ccc_region #   
        gene_distribution_noise = np.zeros((len(noise_cells), cell_vs_gene.shape[1]))
        for j in range (0,  cell_vs_gene.shape[1]):
            gene_distribution_noise[:, j] = np.random.normal(loc=0, scale=3, size = len(noise_cells))
            np.random.shuffle(gene_distribution_noise[:, j])
            
        print('noise: %g to %g'%(np.min(gene_distribution_noise),np.max(gene_distribution_noise) ))
    
    
    for i in range (0, len(noise_cells)):
        cell = noise_cells[i]
        cell_vs_gene[cell, :] = cell_vs_gene[cell, :] + gene_distribution_noise[i,:]
      
#####################################################################





############################
## Add false positives by randomly picking some cells and assigning them expressions from active distribution but without forming pattern ##
## Add false positives by randomly picking some cells and assigning them expressions from active distribution but without forming pattern ##
'''
available_cells = []
for cell in range (0, cell_vs_gene.shape[0]):
    if cell not in active_spot:
        available_cells.append(cell)

np.random.shuffle(available_cells)


for i in range (0, (len(available_cells)*1)//3):
    cell = available_cells[i]
    gene_id = np.arange(lr_gene_count)
    if cell in neighbour_of_actives:
        gene_id = list(set(gene_id)-set(TP_LR_genes)) 
    
    np.random.shuffle(gene_id)
    for j in range (0, (len(gene_id)*1//4)): #
        cell_vs_gene[cell, gene_id[j]] = gene_distribution_active[gene_id[j], cell]
'''

        
##############################
'''
# to reduce number of conections
#cell_vs_gene[:,7] = min_lr_gene_count #-10
#cell_vs_gene[:,15] = min_lr_gene_count #-10
#cell_vs_gene[:,6] = min_lr_gene_count #-10
#cell_vs_gene[:,14] = min_lr_gene_count #-10

available_cells = []
for cell in range (0, cell_vs_gene.shape[0]):
    if cell not in active_spot:
        available_cells.append(cell)

np.random.shuffle(available_cells)
for i in range (0, (len(available_cells)*1)//3):
    cell = available_cells[i]
    gene_id = np.arange(lr_gene_count)
    for j in range (0, (len(gene_id)*2//3)): #
        cell_vs_gene[cell, gene_id[j]] = min_lr_gene_count
'''
##############################
# take quantile normalization.

cell_vs_gene_notNormalized = copy.deepcopy(cell_vs_gene)
temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  #, axis=0
adata_X = np.transpose(temp)  
cell_vs_gene = adata_X
#  cell_vs_gene = copy.deepcopy(cell_vs_gene_notNormalized) #copy.deepcopy(cell_vs_gene_org)



cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    y = sorted(cell_vs_gene[i])
    '''
    y_1 = np.histogram(cell_vs_gene[i])[0] # density: 
    x = range(0, len(y_1))
    kn = KneeLocator(x, y_1, curve='convex', direction='decreasing')
    kn_value = np.histogram(cell_vs_gene[i])[1][kn.knee]    
    '''
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='increasing')
    kn_value = y[kn.knee-1]
    
    cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 99), np.percentile(y, 99) , kn_value])

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
available_edges_to_drop = []
for i in range (0, cell_vs_gene.shape[0]): # ligand                 
    for j in range (0, cell_vs_gene.shape[0]): # receptor
        if dist_X[i,j] <= 0: #distance_matrix[i,j] > threshold_distance:
            continue
        #if i in neighbour_of_actives or j in neighbour_of_actives:
        #    continue
                
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
                        #key = str(i)+'-'+str(j)+str(gene)+'-'+str(gene_rec)
                        #if ligand_dict_dataset[gene][gene_rec] not in lig_rec_dict_TP[i][j]:
                        #    available_edges_to_drop.append([key, communication_scores])
			

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
                    dist.append(cells_ligand_vs_receptor[i][j][k][2])
                    count = count + 1
                    if cells_ligand_vs_receptor[i][j][k][2]>max_score:
                        max_score=cells_ligand_vs_receptor[i][j][k][2]
                    if cells_ligand_vs_receptor[i][j][k][2]<min_score:
                        min_score=cells_ligand_vs_receptor[i][j][k][2] 
                    flag_found=1
                    break
            #if flag_found==1:
                
print('P_class=%d, found=%d, %g, %g, %g'%(P_class, count, min_score, max_score, np.std(dist)))


#################

ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((20))
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
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc, cells_ligand_vs_receptor[i][j][k][3] ])
                    lig_rec.append(cells_ligand_vs_receptor[i][j][k][3])
                if max_local < count_local:
                    max_local = count_local
            '''       
            else: #elif i in neighbour_of_actives and j in neighbour_of_actives:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', '']),
        	'''
                
            ''' '''
            #local_list[count_local] = local_list[count_local] + 1


		
print('len row col %d'%len(row_col))
print('max local %d'%max_local) 
#print('random_activation %d'%len(random_activation_index))
#print('ligand_cells %d'%len(ligand_cells))
print('P_class %d'%P_class) 

options = 'dt-'+datatype+'_lrc'+str(len(lr_database))+'_cp'+str(cell_percent)+'_noise'+str(noise_percent)#'_close'
if noise_add == 1:
    options = options + '_lowNoise'
if noise_add == 2:
    options = options + '_heavyNoise'

total_cells = len(temp_x)

options = options+ '_' + active_type + '_' + distance_measure  + '_cellCount' + str(total_cells)

#options = options + '_f'
options = options + '_3dim' + '_3patterns'+'_temp'+'_sample'+str(sample_no)
#options = options + '_scaled'


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


#options = options+ '_' + 'wFeature'
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options +'_'+'quantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lr_database, lig_rec_dict_TP], fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'quantileTransformed_communication_scores', 'wb') as fp: #b, b_1, a
    pickle.dump(cells_ligand_vs_receptor, fp) #a - [0:5]
    
    
'''   

'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'cellvsgene', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'wb') as fp:
    pickle.dump(cell_vs_gene_notNormalized, fp)

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_communication_scores', 'wb') as fp: #b, b_1, a
#    pickle.dump(cells_ligand_vs_receptor, fp) #a - [0:5]
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options, 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec], fp)


edge_list = []
lig_rec_list = []
row_col_list = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    k = lig_rec[index]
    if edge_weight[index][1] > 0:
        edge_list.append([edge_weight[index][0], edge_weight[index][1], k])
        lig_rec_list.append(k)
        row_col_list.append([i,j])
    
edge_weight = edge_list
row_col = row_col_list
lig_rec = lig_rec_list

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options+'_3dim', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec], fp)

random_activation = []
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([lr_database, lig_rec_dict_TP, random_activation], fp)

############################################################
lig_rec_dict_TP_new = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size): 
    lig_rec_dict_TP_new.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict_TP_new[i].append([])   
        lig_rec_dict_TP_new[i][j] = []

for i in lig_rec_dict_TP:
    for j in lig_rec_dict_TP[i]:
        for k in range (0, len(lig_rec_dict_TP[i][j])):
            lig_rec_dict_TP_new[i][j].append(lig_rec_dict_TP[i][j][k])


lig_rec_dict_TP = copy.deepcopy(lig_rec_dict_TP_new)
lig_rec_dict_TP_new = 0
############################################################## 
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'wb') as fp:
    pickle.dump([temp_x, temp_y, ccc_region], fp)

cell_vs_gene = cell_vs_gene[:,0:lr_gene_count]
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'cellvslrgene', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)

''' 

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

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'rb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_cellvsgene', 'rb') as fp: #'not_quantileTransformed'
    cell_vs_gene = pickle.load(fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'rb') as fp:
    temp_x, temp_y, ccc_region  = pickle.load(fp)

data_list_pd = pd.DataFrame(temp_x)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_cell_'+options+'_x.csv', index=False, header=False)
data_list_pd = pd.DataFrame(temp_y)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_cell_'+options+'_y.csv', index=False, header=False)


data_list=defaultdict(list)
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[1]):
        data_list['a-'+str(i)].append(cell_vs_gene[i][j]) #(cell_vs_gene[i][j]-min_value)/(max_value-min_value)
        
        
data_list_pd = pd.DataFrame(data_list)    
gene_name = []
for i in range (0, cell_vs_gene.shape[1]):
    gene_name.append('g'+str(i))
    
data_list_pd[' ']=gene_name   
data_list_pd = data_list_pd.set_index(' ')    
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_'+options+'_not_quantileTransformed.csv')

data_list=dict()
data_list['ligand']=[]
data_list['receptor']=[]
for i in range (0, len(lr_database)):
    data_list['ligand'].append('g'+str(lr_database[i][0]))
    data_list['receptor'].append('g'+str(lr_database[i][1]))
    
data_list_pd = pd.DataFrame(data_list)        
data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_lr_'+options+'.csv', index=False)
	
	
###############
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'notQuantileTransformed', 'rb') as fp:
    cell_vs_gene = pickle.load(fp
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
    plot_dict['Type'].append('naive_model_HeavyNoise')

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'naive_model', 'wb') as fp: #b, b_1, a
    pickle.dump(plot_dict, fp) #a - [0:5]

###########################################   
plot_dict = defaultdict(list)
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 5
#csv_record_dict = defaultdict(list)
for run_time in range (0,total_runs):
    run = run_time
    #if run in [1, 2, 4, 7, 8]:
    #    continue

    X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_1pattern_f_tanh_3d_temp_'+filename[run]+'_attention_l1.npy' #split_ #dropout_
    X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) #4patterns_knn10
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

####################
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 10
percentage_threshold = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
plot_dict_list = []
#plot_dict_list.append(defaultdict(list))
#plot_dict_list.append(defaultdict(list))
#plot_dict_list.append(defaultdict(list))
#plot_dict_list.append(defaultdict(list))
#plot_dict_list.append(defaultdict(list))
for run_time in range (0,total_runs):
    plot_dict_list.append(defaultdict(list))
    run = run_time
    print('run %d'%run)
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_6_path_knn10_f_3d_'+filename[run]+'_attention_l1.npy'
    X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_tanh_3d_temp_'+filename[run]+'_attention_l1.npy' #split_ #dropout_
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_tanh_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3 #_swappedLRid
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_relu_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_tanh_3dim_dropout_'+filename[run]+'_attention_l1.npy' #withFeature_4_pattern_overlapped_highertail, tp7p_,4_pattern_differentLRs, tp7p_broad_active, 4_r3,5_close, overlap_noisy, 6_r3
    X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 
    # [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized, X_attention_score_unnormalized_l1, X_attention_score_normalized]
    csv_record_dict = defaultdict(list)
    
    for percentage_value in percentage_threshold:
        for l in [2, 3]:
            #l=3 #2 ## 
            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                distribution.append(X_attention_bundle[l][index][0])
    
            max_value = np.max(distribution)
            min_value = np.min(distribution)
            
			
				
            attention_scores = []
            datapoint_size = temp_x.shape[0]
            for i in range (0, datapoint_size):
                attention_scores.append([])   
                for j in range (0, datapoint_size):	
                    attention_scores[i].append([])   
                    attention_scores[i][j] = []
                    
            min_attention_score = max_value
            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
                #    continue
                scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
                attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
                if min_attention_score > scaled_score:
                    min_attention_score = scaled_score
                distribution.append(scaled_score)
				
            print('min attention score with scaling %g'%min_attention_score)
            #######################
            #plt.hist(distribution, color = 'blue', bins = int(len(distribution)/5))
            save_path = '/cluster/home/t116508uhn/64630/'
            #plt.savefig(save_path+'distribution_type6_f_3d_tanh_'+filename[run]+'.svg', dpi=400)
            plt.clf()
            
        
            datapoint_size = temp_x.shape[0]
            count = 0
            existing_lig_rec_dict = []
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
                            key_value = str(i) +'-'+ str(j) + '-' + str(lig_rec_dict[i][j][k])
                            csv_record_dict[key_value].append([attention_scores[i][j][k], run])
                            count = count + 1
                            #distribution_partial.append(attention_scores[i][j][k])

    ############### merge multiple runs ##################
        for key_value in csv_record_dict.keys():
            run_dict = defaultdict(list)
            for scores in csv_record_dict[key_value]:
                run_dict[scores[1]].append(scores[0])
        
            for runs in run_dict.keys():
                run_dict[runs] = np.mean(run_dict[runs])
        
        
            csv_record_dict[key_value] = []
            for runs in run_dict.keys():
                csv_record_dict[key_value].append([run_dict[runs],runs]) # this has values for all the edges for all runs
    

    
    #######################################
        csv_record_intersect_dict = defaultdict(list)
        for key_value in csv_record_dict.keys():
            if len(csv_record_dict[key_value])>=1: #3: #((total_runs*80)//100):
                score = 0
                for k in range (0, len(csv_record_dict[key_value])):
                    score = score + csv_record_dict[key_value][k][0]
                score = score/len(csv_record_dict[key_value]) # take the average score
    
                csv_record_intersect_dict[key_value].append(score)
    
    ########################################
        existing_lig_rec_dict = []
        for i in range (0, datapoint_size):
            existing_lig_rec_dict.append([])   
            for j in range (0, datapoint_size):	
                existing_lig_rec_dict[i].append([])   
                existing_lig_rec_dict[i][j] = []    
                
        for key_value in csv_record_intersect_dict.keys():
            item = key_value.split('-')
            i = int(item[0])
            j = int(item[1])
            LR_pair_id = int(item[2])
            existing_lig_rec_dict[i][j].append(LR_pair_id)
        #######################################
        confusion_matrix = np.zeros((2,2))
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):
    
                if i==j: 
                    continue
    
                if len(existing_lig_rec_dict[i][j])>0:
                    for k in existing_lig_rec_dict[i][j]:   
                        
                        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                            #print(k)
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
        plot_dict_list[run]['FPR'].append(FPR_value)
        plot_dict_list[run]['TPR'].append(TPR_value)
        plot_dict_list[run]['Type'].append('run_'+str(run+1))


####################


FPR_list = []
TPR_list = []
for i in range (0, len(percentage_threshold)):
    FPR = []
    for run in range (0,total_runs):
        FPR.append(plot_dict_list[run]['FPR'][i])  
    FPR = np.mean(FPR)
    FPR_list.append(FPR)
    
    TPR = []
    for run in range (0,total_runs):
        TPR.append(plot_dict_list[run]['TPR'][i])  
    TPR = np.mean(TPR)
    TPR_list.append(TPR)

plot_dict = defaultdict(list)
for i in range (0, len(percentage_threshold)):
    plot_dict['FPR'].append(FPR_list[i])
    plot_dict['TPR'].append(TPR_list[i])
    plot_dict['Type'].append('NEST_average_10runs')
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'average_10runs', 'wb') as fp: #b, b_1, a
    pickle.dump([plot_dict, plot_dict_list], fp) #a - [0:5]


########################################################################################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'average_5runs', 'rb') as fp: #b, b_1, a
    plot_dict_temp, plot_dict_list_temp = pickle.load(fp) #a - [0:5]

for run in range (0, 5):
    for i in range (0, len(plot_dict_list_temp[run]['Type'])):
        plot_dict_list[run]['FPR'].append(plot_dict_list_temp[run]['FPR'][i])
        plot_dict_list[run]['TPR'].append(plot_dict_list_temp[run]['TPR'][i])
        plot_dict_list[run]['Type'].append(plot_dict_list_temp[run]['Type'][i])  
#########################################################################################################################


plot_dict = defaultdict(list)
for run in range (0, 5):
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append(plot_dict_list_temp[run]['Type'][0])
    for i in range (0, len(plot_dict_list_temp[run]['Type'])):
        plot_dict['FPR'].append(plot_dict_list_temp[run]['FPR'][i])
        plot_dict['TPR'].append(plot_dict_list_temp[run]['TPR'][i])
        plot_dict['Type'].append(plot_dict_list_temp[run]['Type'][i])  
        
data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:Q',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'plot_avg_naive.html')

####################

####################
# ensemble 
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 5
percentage_threshold = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
plot_dict = defaultdict(list)
for percentage_value in percentage_threshold:
    csv_record_dict = defaultdict(list)
    for l in [2, 3]: # 2 = layer 2, 3 = layer 1
        for run_time in range (0,total_runs):
            run = run_time
            X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_tanh_3d_temp_'+filename[run]+'_attention_l1.npy' #split_ #dropout_
            X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                distribution.append(X_attention_bundle[l][index][0])
    
            max_value = np.max(distribution)
            min_value = np.min(distribution)
            
			
				
            attention_scores = []
            datapoint_size = temp_x.shape[0]
            for i in range (0, datapoint_size):
                attention_scores.append([])   
                for j in range (0, datapoint_size):	
                    attention_scores[i].append([])   
                    attention_scores[i][j] = []
                    
            min_attention_score = max_value
            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
                #    continue
                scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
                attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
                if min_attention_score > scaled_score:
                    min_attention_score = scaled_score
                distribution.append(scaled_score)
				
            #print('min attention score with scaling %g'%min_attention_score)
            

            
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



            datapoint_size = temp_x.shape[0]

            count = 0
            existing_lig_rec_dict = []
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
                            key_value = str(i) +'-'+ str(j) + '-' + str(lig_rec_dict[i][j][k])
                            csv_record_dict[key_value].append([attention_scores[i][j][k], run])
                            count = count + 1
                            #distribution_partial.append(attention_scores[i][j][k])


    ############### merge multiple runs ##################
    for key_value in csv_record_dict.keys():
        run_dict = defaultdict(list)
        for scores in csv_record_dict[key_value]:
            run_dict[scores[1]].append(scores[0])

        for runs in run_dict.keys():
            run_dict[runs] = np.mean(run_dict[runs])


        csv_record_dict[key_value] = []
        for runs in run_dict.keys():
            csv_record_dict[key_value].append([run_dict[runs],runs])


    
    #######################################

    
    csv_record_intersect_dict = defaultdict(list)
    for key_value in csv_record_dict.keys():
        if len(csv_record_dict[key_value])>=4: #total_runs 3: #((total_runs*80)/100):
            score = 0
            for k in range (0, len(csv_record_dict[key_value])):
                score = score + csv_record_dict[key_value][k][0]
            score = score/len(csv_record_dict[key_value]) # take the average score

            csv_record_intersect_dict[key_value].append(score)
    
    ########################################
    existing_lig_rec_dict = []
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []    
            
    for key_value in csv_record_intersect_dict.keys():
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        LR_pair_id = int(item[2])
        existing_lig_rec_dict[i][j].append(LR_pair_id)
    #######################################
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue

            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print(k)
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
    #plot_dict['Type'].append('ensemble_100percent')
    plot_dict['Type'].append('ensemble_80percent')

#plt.hist(distribution_partial, color = 'blue', bins = int(len(distribution_partial)/5))
#save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'distribution_e_3d_relu_partial_'+filename[run]+'_'+str(percentage_value)+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_gatconv_'+filename[run]+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_tanh_'+filename[run]+'.svg', dpi=400)
#plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'ensemble_80percent', 'wb') as fp: #b, b_1, a
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'ensemble_100percent', 'wb') as fp: #b, b_1, a
    pickle.dump(plot_dict, fp) #a - [0:5]

######### rank product ####
#filename = ["r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20"]
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 10
plot_dict = defaultdict(list)
distribution_rank = []
all_edge_sorted_by_avgrank = []
for layer in range (0, 2):
    distribution_rank.append([])
    all_edge_sorted_by_avgrank.append([])

layer = -1
percentage_value = 0

for l in [2, 3]: # 2 = layer 2, 3 = layer 1
    layer = layer + 1
    csv_record_dict = defaultdict(list)
    for run_time in range (0,total_runs):
        run = run_time
        X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_relu_3d_lowNoise_temp_'+filename[run]+'_attention_l1.npy'
        #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_relu_3d_temp_'+filename[run]+'_attention_l1.npy'
        #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_tanh_3d_temp_sample3_'+filename[run]+'_attention_l1.npy' #split_ #dropout_
        X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) # f_

        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            distribution.append(X_attention_bundle[l][index][0])

        max_value = np.max(distribution)
        min_value = np.min(distribution)
        
        
            
        attention_scores = []
        datapoint_size = temp_x.shape[0]
        for i in range (0, datapoint_size):
            attention_scores.append([])   
            for j in range (0, datapoint_size):	
                attention_scores[i].append([])   
                attention_scores[i][j] = []
                
        min_attention_score = max_value
        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
            #    continue
            scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
            attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
            if min_attention_score > scaled_score:
                min_attention_score = scaled_score
            distribution.append(scaled_score)
            
        #print('min attention score with scaling %g'%min_attention_score)



        
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



        datapoint_size = temp_x.shape[0]

        count = 0
        existing_lig_rec_dict = []
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
                        c10830cc_index_dict[j] = ''
                        existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                        key_value = str(i) +'-'+ str(j) + '-' + str(lig_rec_dict[i][j][k])
                        csv_record_dict[key_value].append([attention_scores[i][j][k], run])
                        count = count + 1
                        #distribution_partial.append(attention_scores[i][j][k])


############### merge multiple runs ##################
    for key_value in csv_record_dict.keys():
        run_dict = defaultdict(list)
        for scores in csv_record_dict[key_value]:
            run_dict[scores[1]].append(scores[0])

        for runs in run_dict.keys():
            run_dict[runs] = np.mean(run_dict[runs])


        csv_record_dict[key_value] = []
        for runs in run_dict.keys():
            csv_record_dict[key_value].append([run_dict[runs],runs])


    
    #######################################
    
    all_edge_list = []
    for key_value in csv_record_dict.keys():
        edge_score_runs = []
        edge_score_runs.append(key_value)
        for runs in csv_record_dict[key_value]:
            edge_score_runs.append(runs[0]) # 
            
        all_edge_list.append(edge_score_runs)

    ## Find the rank of product
    edge_rank_dictionary = defaultdict(list)
    # sort the all_edge_list by runs and record the rank
    print('total runs %d'%total_runs)
    for runs in range (0, total_runs):
        sorted_list_temp = sorted(all_edge_list, key = lambda x: x[runs+1], reverse=True) # sort based on score by current run and large to small
        for rank in range (0, len(sorted_list_temp)):
            edge_rank_dictionary[sorted_list_temp[rank][0]].append(rank+1) # small rank being high attention

    all_edge_avg_rank = []
    for key_val in edge_rank_dictionary.keys():
        rank_product = 1
        for i in range (0, len(edge_rank_dictionary[key_val])):
            rank_product = rank_product * edge_rank_dictionary[key_val][i]
            
        all_edge_avg_rank.append([key_val, rank_product**(1/total_runs)])  # small rank being high attention
        distribution_rank[layer].append(rank_product**(1/total_runs))
        
    all_edge_sorted_by_avgrank[layer] = sorted(all_edge_avg_rank, key = lambda x: x[1]) # small rank being high attention 

# now you can start roc curve by selecting top 90%, 80%, 70% edges ...so on

percentage_value = 10
percentage_threshold = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for percentage_value in percentage_threshold:
    csv_record_intersect_dict = defaultdict(list)
    for layer in range (0, 2):
        threshold_up = np.percentile(distribution_rank[layer], percentage_value)
        for i in range (0, len(all_edge_sorted_by_avgrank[layer])):
            if all_edge_sorted_by_avgrank[layer][i][1] <= threshold_up:
                csv_record_intersect_dict[all_edge_sorted_by_avgrank[layer][i][0]].append(all_edge_sorted_by_avgrank[layer][i][1])
    '''
    threshold_up = np.percentile(distribution_rank_layer2, percentage_value)
    for i in range (0, len(all_edge_sorted_by_avgrank_layer2)):
        if all_edge_sorted_by_avgrank_layer2[i][1] <= threshold_up:
            csv_record_intersect_dict[all_edge_sorted_by_avgrank_layer2[i][0]].append(all_edge_sorted_by_avgrank_layer2[i][1])
    '''
    ###### this small block does not have any impact now ###########
    for key_value in csv_record_intersect_dict.keys():  
       if len(csv_record_intersect_dict[key_value])>1:
           csv_record_intersect_dict[key_value] = [np.mean(csv_record_intersect_dict[key_value])]
    #######################################################
    
    existing_lig_rec_dict = []
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []    
            
    for key_value in csv_record_intersect_dict.keys():
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        LR_pair_id = int(item[2])
        existing_lig_rec_dict[i][j].append(LR_pair_id)
    #######################################
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue

            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print(k)
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
    plot_dict['Type'].append('rank_product') #_lowNoise #_heavyNoise #_relu

#plt.hist(distribution_partial, color = 'blue', bins = int(len(distribution_partial)/5))
#save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'distribution_e_3d_relu_partial_'+filename[run]+'_'+str(percentage_value)+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_gatconv_'+filename[run]+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_tanh_'+filename[run]+'.svg', dpi=400)
#plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'rank_product_relu_10runs', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict, fp) #a - [0:5]

########### z score ################################################################
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 10
z_significance = 1.65
plot_dict = defaultdict(list)
distribution_rank = []
all_edge_sorted_by_avgrank = []
for layer in range (0, 2):
    distribution_rank.append([])
    all_edge_sorted_by_avgrank.append([])

layer = -1
percentage_value = 0

for l in [2, 3]: # 2 = layer 2, 3 = layer 1
    csv_record_dict = defaultdict(list)
    layer = layer + 1
    for run_time in range (0,total_runs):
        run = run_time
 
        X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_roc_control_model_uniform_path_th4_lrc112_cell5000_tanh_3d_temp_'+filename[run]+'_attention_l1.npy' #split_ #dropout_
        X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            distribution.append(X_attention_bundle[l][index][0])

        max_value = np.max(distribution)
        min_value = np.min(distribution)
        
        
            
        attention_scores = []
        datapoint_size = temp_x.shape[0]
        for i in range (0, datapoint_size):
            attention_scores.append([])   
            for j in range (0, datapoint_size):	
                attention_scores[i].append([])   
                attention_scores[i][j] = []
                
        min_attention_score = max_value
        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
            #    continue
            scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
            attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
            if min_attention_score > scaled_score:
                min_attention_score = scaled_score
            distribution.append(scaled_score)
            
        #print('min attention score with scaling %g'%min_attention_score)



        
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



        datapoint_size = temp_x.shape[0]

        count = 0
        existing_lig_rec_dict = []
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
                        key_value = str(i) +'-'+ str(j) + '-' + str(lig_rec_dict[i][j][k])
                        csv_record_dict[key_value].append([attention_scores[i][j][k], run])
                        count = count + 1
                        #distribution_partial.append(attention_scores[i][j][k])


############### merge multiple runs ##################

        
    for key_value in csv_record_dict.keys():
        run_dict = defaultdict(list)
        for scores in csv_record_dict[key_value]:
            run_dict[scores[1]].append(scores[0])

        for runs in run_dict.keys():
            run_dict[runs] = np.mean(run_dict[runs])


        csv_record_dict[key_value] = []
        for runs in run_dict.keys():
            csv_record_dict[key_value].append([run_dict[runs],runs])


    # for each run, keep only those runs whose z score is significant
    key_list = []
    score_list = []
    for i in range (0, total_runs):
        score_list.append([])
        
    for key_value in csv_record_dict.keys():
        key_list.append(key_value)
        for i in range (0, len(csv_record_dict[key_value])):
            runs = csv_record_dict[key_value][i][1]
            score_list[runs].append(csv_record_dict[key_value][i][0])
        
    for runs in range (0, total_runs):
        z_score_list = scipy.stats.zscore(score_list[runs]) 
        score_list[runs] = z_score_list

    
    for i in range (0, len(key_list)):
        count_of_significance = []
        for runs in range (0, total_runs):
            if np.abs(score_list[runs][i]) >= z_significance:
                count_of_significance.append(np.abs(score_list[runs][i]))

        if len(count_of_significance) >= 1:
            csv_record_dict[key_list[i]] = np.mean(count_of_significance) 
        else:
            csv_record_dict[key_list[i]] = 0
    #######################################
 
    all_edge_avg_rank = []
    for key_value in csv_record_dict.keys():
        distribution_rank[layer].append(csv_record_dict[key_value])
        all_edge_avg_rank.append([key_value, csv_record_dict[key_value]]) # high score is high attention
        
    all_edge_sorted_by_avgrank[layer] = sorted(all_edge_avg_rank, key = lambda x: x[1], reverse = True) 

# now you can start roc curve by selecting top 90%, 80%, 70% edges ...so on


percentage_threshold = [90, 80, 70, 60, 50, 40, 30, 20, 10]
for percentage_value in percentage_threshold:
    csv_record_intersect_dict = defaultdict(list)
    for layer in range (0, 2):
        threshold_down = np.percentile(distribution_rank[layer], percentage_value)
        for i in range (0, len(all_edge_sorted_by_avgrank[layer])):
            if all_edge_sorted_by_avgrank[layer][i][1] >= threshold_down:
                csv_record_intersect_dict[all_edge_sorted_by_avgrank[layer][i][0]].append(all_edge_sorted_by_avgrank[layer][i][1])
    '''
    threshold_up = np.percentile(distribution_rank_layer2, percentage_value)
    for i in range (0, len(all_edge_sorted_by_avgrank_layer2)):
        if all_edge_sorted_by_avgrank_layer2[i][1] <= threshold_up:
            csv_record_intersect_dict[all_edge_sorted_by_avgrank_layer2[i][0]].append(all_edge_sorted_by_avgrank_layer2[i][1])
    '''
    ###### this small block does not have any impact now ###########
    for key_value in csv_record_intersect_dict.keys():  
       if len(csv_record_intersect_dict[key_value])>1:
           csv_record_intersect_dict[key_value] = [np.mean(csv_record_intersect_dict[key_value])]
    #######################################################
    
    existing_lig_rec_dict = []
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []    
            
    for key_value in csv_record_intersect_dict.keys():
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        LR_pair_id = int(item[2])
        existing_lig_rec_dict[i][j].append(LR_pair_id)
    #######################################
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            if i==j: 
                continue

            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print(k)
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
    plot_dict['Type'].append('z_score')

#plt.hist(distribution_partial, color = 'blue', bins = int(len(distribution_partial)/5))
#save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'distribution_e_3d_relu_partial_'+filename[run]+'_'+str(percentage_value)+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_gatconv_'+filename[run]+'.svg', dpi=400)
#plt.savefig(save_path+'distribution_e_3d_tanh_'+filename[run]+'.svg', dpi=400)
#plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'rank_product', 'wb') as fp: #b, b_1, a
    pickle.dump(plot_dict, fp) #a - [0:5]

##############



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
###########################


datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if i in ccc_index_dict:
        datapoint_label.append(1)
    else:
        datapoint_label.append(0)

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
######################################################## Niches ######################################################################################################################################

# get all the edges and their scaled scores that they use for plotting the heatmap
df_pair_vs_cells = pd.read_csv('/cluster/home/t116508uhn/niches_output_pair_vs_cells_'+options+'.csv')

edge_pair_dictionary = defaultdict(dict) # edge_pair_dictionary[edge[pair]]=score
coexpression_scores = []
lig_rec_dict_all = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    coexpression_scores.append([])   
    lig_rec_dict_all.append([])   
    for j in range (0, datapoint_size):	
        coexpression_scores[i].append([])   
        coexpression_scores[i][j] = []
        lig_rec_dict_all[i].append([])   
        lig_rec_dict_all[i][j] = []

distribution_all = []
for col in range (1, len(df_pair_vs_cells.columns)):
    col_name = df_pair_vs_cells.columns[col]
    l_c = df_pair_vs_cells.columns[col].split("—")[0]
    r_c = df_pair_vs_cells.columns[col].split("—")[1]
    l_c = l_c.split('.')[1]
    r_c = r_c.split('.')[1]
    i = int(l_c)
    j = int(r_c)
    
    for index in range (0, len(df_pair_vs_cells.index)):
        lig_rec_dict_all[i][j].append(df_pair_vs_cells.index[index])
        coexpression_scores[i][j].append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])
        distribution_all.append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])
        edge_pair_dictionary[str(i)+'-'+str(j)][df_pair_vs_cells.index[index]]=df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]]


######### read which edge belongs to which cluster type #############################
vector_type = pd.read_csv('/cluster/home/t116508uhn/niches_VectorType_'+options+'.csv')
clusterType_edge_dictionary = defaultdict(list)
for index in range (0, len(vector_type.index)):
    cell_cell_pair = vector_type['Unnamed: 0'][index]
    l_c = cell_cell_pair.split("—")[0]
    r_c = cell_cell_pair.split("—")[1]
    l_c = l_c.split('.')[1]
    r_c = r_c.split('.')[1]
    i = int(l_c)
    j = int(r_c)

    cluster_type = vector_type['VectorType'][index]
    clusterType_edge_dictionary[cluster_type].append(str(i)+'-'+str(j))
    
######## read the top5 edges (ccc) by Niches ########################################
attention_scores_temp = []
lig_rec_dict_temp = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores_temp.append([])   
    lig_rec_dict_temp.append([])   
    for j in range (0, datapoint_size):	
        attention_scores_temp[i].append([])   
        attention_scores_temp[i][j] = []
        lig_rec_dict_temp[i].append([])   
        lig_rec_dict_temp[i][j] = []
        

marker_list = pd.read_csv('/cluster/home/t116508uhn/niches_output_ccc_lr_pairs_markerList_top5_'+options+'.csv')
marker_list = marker_list.sort_values(by=['myAUC'], ascending=False) #marker_list.sort_values(by=['avg_log2FC'], ascending=False) # high fc to low fc
positive_class_found = 0
distribution_temp = []
total_edge_count = 0
flag_break = 0
for index in range (0, len(marker_list.index)):
    cluster_type = marker_list['cluster'][index]
    pair_type = marker_list['gene'][index]
    ligand_gene = pair_type.split('—')[0]
    receptor_gene = pair_type.split('—')[1]
    ligand_gene = int(ligand_gene.split('g')[1])
    receptor_gene = int(receptor_gene.split('g')[1])
    lr_pair_id = ligand_dict_dataset[ligand_gene][receptor_gene] 
    #if lr_pair_id>12: 
    #    continue
    edge_list = clusterType_edge_dictionary[cluster_type]
    for edge in edge_list:
        ccc_score_scaled = edge_pair_dictionary[edge][lr_pair_id]
        i = int(edge.split('-')[0])
        j = int(edge.split('-')[1])
        total_edge_count = total_edge_count + 1
        if total_edge_count > len(row_col):
            flag_break = 1
            break

        lig_rec_dict_temp[i][j].append(lr_pair_id)
        attention_scores_temp[i][j].append(ccc_score_scaled)
        distribution_temp.append(ccc_score_scaled)
	    
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and lr_pair_id in lig_rec_dict_TP[i][j]:
            positive_class_found = positive_class_found + 1
	
    if flag_break == 1:
        break

print('positive_class_found %d'%positive_class_found)    
lig_rec_dict = lig_rec_dict_temp
attention_scores = attention_scores_temp
distribution = distribution_temp
negative_class = len(distribution) - positive_class_found

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

    ccc_index_dict = dict()
    threshold_down =  np.percentile(sorted(distribution), percentage_value)
    threshold_up =  np.percentile(sorted(distribution), 100)
    connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
    rec_dict = defaultdict(dict)
    total_edges_count = 0
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            #if i==j: 
            #    continue
            atn_score_list = attention_scores[i][j]
            #print(len(atn_score_list))
            
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                    total_edges_count = total_edges_count + 1
                    


    ############# 
    print('total edges %d'%total_edges_count)
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            #if i==j: 
            #    continue
            ''' 
            if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i]:
                for k in range (0, len(lig_rec_dict_TP[i][j])):
                    if lig_rec_dict_TP[i][j][k] in existing_lig_rec_dict[i][j]: #
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[0][1] = confusion_matrix[0][1] + 1 

            '''
            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print("i=%d j=%d k=%d"%(i, j, k))
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
             
    print('%d, %g, %g'%(percentage_value,  (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    FPR_value = (confusion_matrix[1][0]/negative_class)#*100
    TPR_value = (confusion_matrix[0][0]/positive_class)#*100
    plot_dict['FPR'].append(FPR_value)
    plot_dict['TPR'].append(TPR_value)
    plot_dict['Type'].append('Niches') #_lowNoise


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'Niches', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict, fp) #a - [0:5]

######################### COMMOT ###############################################################################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_'+options+'_commot_result', 'rb') as fp:
    attention_scores, lig_rec_dict, distribution = pickle.load(fp)            


distribution = sorted(distribution, reverse=True)
#distribution = distribution[0:len(row_col)] # len(distribution) = 6634880, len(row_col)=21659
#negative_class=len(distribution)-confusion_matrix[0][0]

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

    ccc_index_dict = dict()
    threshold_down =  np.percentile(distribution, percentage_value)
    threshold_up =  np.percentile(distribution, 100)
    connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
    rec_dict = defaultdict(dict)
    total_edges_count = 0
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            atn_score_list = attention_scores[i][j]
            
            
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                    total_edges_count = total_edges_count + 1
                    


    ############# 
    print('total edges %d'%total_edges_count)
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            #if i==j: 
            #    continue
            ''' 
            if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i]:
                for k in range (0, len(lig_rec_dict_TP[i][j])):
                    if lig_rec_dict_TP[i][j][k] in existing_lig_rec_dict[i][j]: #
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[0][1] = confusion_matrix[0][1] + 1 

            '''
            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print("i=%d j=%d k=%d"%(i, j, k))
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
             
    print('%d, %g, %g'%(percentage_value,  (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    FPR_value = (confusion_matrix[1][0]/negative_class)#*100
    TPR_value = (confusion_matrix[0][0]/positive_class)#*100
    plot_dict['FPR'].append(FPR_value)
    plot_dict['TPR'].append(TPR_value)
    plot_dict['Type'].append('COMMOT') #_lowNoise


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'COMMOT', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict_temp, fp) #a - [0:5]


######################### PLOTS ################################################################################################################
datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if i in ccc_index_dict:
        datapoint_label.append(1)
    else:
        datapoint_label.append(0)

data_list=dict()
data_list['X']=[]
data_list['Y']=[]   
data_list['label']=[]  
for i in range (0, len(temp_x)):    
    data_list['X'].append(temp_x[i])
    data_list['Y'].append(temp_y[i])    
    data_list['label'].append(datapoint_label[i]) 

data_list_pd = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", len(list(set(data_list['label']))) )
chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
    	
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    color=alt.Color('label:N', scale=alt.Scale(range=set1)),
)
chart.save('/cluster/home/t116508uhn/' +'input_data.html')

##############################################################
sample_type = ["", "_LowNoise", "_HighNoise"]
sample_name = ["dt-path_uniform_distribution_lrc112_cp100_noise0_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp", 
              "dt-path_uniform_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp",
              "dt-path_uniform_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp_v2"]

for t in range (1, 2): #len(sample_name)):
    plot_dict = defaultdict(list)
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t] +'_'+'naive_model', 'rb') as fp: #b, b_1, a
        plot_dict_temp = pickle.load(fp) #a - [0:5]
    
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append("Naive"+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append("Naive"+sample_type[t]) #(plot_dict_temp['Type'][i])
    ###
    
    ######
    #with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'rank_product_lowNoise_10runs', 'rb') as fp: #b, b_1, a
    #with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'rank_product_11to20runs', 'rb') as fp: #b, b_1, a
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t] +'_'+'rank_product_10runs', 'rb') as fp: #b, b_1, a
        plot_dict_temp = pickle.load(fp) #a - [0:5]
    
    plot_dict_temp['FPR'].append(1.0)
    plot_dict_temp['TPR'].append(1.0)
    plot_dict_temp['Type'].append(plot_dict_temp['Type'][1])
    
    
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append("NEST"+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append("NEST"+sample_type[t]) #(plot_dict_temp['Type'][i])
    
    ######
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t] +'_'+'rank_product_relu_10runs', 'rb') as fp: #b, b_1, a
        plot_dict_temp = pickle.load(fp) #a - [0:5]
    
#    plot_dict_temp['FPR'].append(1.0)
#    plot_dict_temp['TPR'].append(1.0)
#    plot_dict_temp['Type'].append(plot_dict_temp['Type'][1])
    
    
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append("NEST_ReLU"+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append("NEST_ReLU"+sample_type[t]) #(plot_dict_temp['Type'][i])
    
    ######
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t]  +'_'+'Niches', 'rb') as fp: #b, b_1, a
        plot_dict_temp = pickle.load(fp) #a - [0:5]
        
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append('Niches'+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append('Niches'+sample_type[t]) #(plot_dict_temp['Type'][i])
    
    ######
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t]  +'_'+'COMMOT', 'rb') as fp: #b, b_1, a
        plot_dict_temp = pickle.load(fp) #a - [0:5]
        
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append('COMMOT'+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append('COMMOT'+sample_type[t]) #(plot_dict_temp['Type'][i])
    
        
    
    data_list_pd = pd.DataFrame(plot_dict)    
    chart = alt.Chart(data_list_pd).mark_line().encode(
        x='FPR:Q',
        y='TPR:Q',
        color='Type:N',
    )	
    save_path = '/cluster/home/t116508uhn/'
    chart.save(save_path+'plot_uniform'+sample_type[t]+'.html')

################################################################################################################################
niches_FPR = [0, .08, .15, .25, .35, .43, .52, .62, .72, .83, 1.00]
niches_TPR = [0, .20, .43, .51, .61, .76, .89, .97, .98, .99, 1.00]
for i in range (0, 11):
    plot_dict['FPR'].append(niches_FPR[i])
    plot_dict['TPR'].append(niches_TPR[i])
    plot_dict['Type'].append('Niches')
    
'''    
niches_FPR = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.00]
niches_TPR = [0, .43097, .451493, .455224, .91791, .91791, .91791, .91791, .91791, .91791, .91791]
for i in range (0, 11):
    plot_dict['FPR'].append(niches_FPR[i])
    plot_dict['TPR'].append(niches_TPR[i])
    plot_dict['Type'].append('Niches with 20 nearest neighbour')
    
######
COMMOT_FPR = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.00]
COMMOT_TPR = [0, .50, .50, .50, .50, .50, .50, .50, .50, .50, .50]
for i in range (0, 11):
    plot_dict['FPR'].append(COMMOT_FPR[i])
    plot_dict['TPR'].append(COMMOT_TPR[i])
    plot_dict['Type'].append('COMMOT')        
'''
data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:Q',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/'
chart.save(save_path+'plot_uniform.html') # _lowNoise
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t] +'_'+'average_10runs', 'rb') as fp: #b, b_1, a
    plot_dict_temp, plot_dict_list_temp= pickle.load(fp) #a - [0:5]
	
plot_dict['FPR'].append(0)
plot_dict['TPR'].append(0)
plot_dict['Type'].append(plot_dict_temp['Type'][0])
for i in range (0, len(plot_dict_temp['Type'])):
    plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
    plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
    plot_dict['Type'].append(plot_dict_temp['Type'][i])
###
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + sample_name[t] +'_'+'ensemble_100percent', 'rb') as fp: #b, b_1, a
    plot_dict_temp = pickle.load(fp) #a - [0:5]
plot_dict['FPR'].append(0)
plot_dict['TPR'].append(0)
plot_dict['Type'].append(plot_dict_temp['Type'][0])
for i in range (0, len(plot_dict_temp['Type'])):
    plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
    plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
    plot_dict['Type'].append(plot_dict_temp['Type'][i])
           
###
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'ensemble_80percent', 'rb') as fp: #b, b_1, a
    plot_dict_temp = pickle.load(fp) #a - [0:5]
plot_dict['FPR'].append(0)
plot_dict['TPR'].append(0)
plot_dict['Type'].append(plot_dict_temp['Type'][0])
for i in range (0, len(plot_dict_temp['Type'])):
    plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
    plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
    plot_dict['Type'].append(plot_dict_temp['Type'][i])
           
##############################################################
