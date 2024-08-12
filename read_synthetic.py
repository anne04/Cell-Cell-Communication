import os
#import glob
import pandas as pd
import copy
import csv
import numpy as np
import sys
from collections import defaultdict
import stlearn as st
import scipy
import pickle
import gzip




with gzip.open('uniform_distribution_cellvsgene_not_normalized', 'rb') as fp:
    cell_vs_gene = pickle.load(fp)
  
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[1]):
        print(cell_vs_gene[i][j])
#################################################################################################################################################################################
with gzip.open('uniform_distribution_coordinate', 'rb') as fp:
    temp_x, temp_y, no_need  = pickle.load(fp)

coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
 
datapoint_size = temp_x.shape[0] 
#####################################################################################################################
with gzip.open("uniform_distribution_ground_truth_ccc" , 'rb') as fp:            
    no_need1, lig_rec_dict_TP, no_need2 = pickle.load( fp

# lig_rec_dict_TP is a dict(dict(list of integers))
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
            for k in range (0, len(lig_rec_dict_TP[i][j])): # lig_rec_dict_TP[i][j] is a list of lr pairs from cell i to j. The lr name or id = integers 
                print('lr pair %d exist from cell i=%d to cell j=%d'%(lig_rec_dict_TP[i][j][k], i, j))

###############################################################################################################
with gzip.open("uniform_distribution_input_graph" , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
    row_col, edge_weight, lig_rec  = pickle.load(fp)  

# row_col is a list of pairs: [[i, j], [k, l], ...] where each pair is an edge in the input graph
for index in range (0, len(row_col)):
    from_cell = row_col[index][0]
    to_cell = row_col[index][1]

# lig_rec is a list of pairs name/ids in the same order as above. 
for index in range (0, len(lig_rec)):
    lr_id = lig_rec[index] # between i to j when index = 0. between k to l when index = 1
    
#############################################################################################################
		
