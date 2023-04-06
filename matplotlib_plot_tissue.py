import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd
import gzip
from kneed import KneeLocator
import copy 
import altairThemes
import altair as alt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()

spot_diameter = 89.43 #pixels
############

############

 
####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
sc.pp.filter_genes(adata_h5, min_cells=1)
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
###############################################
total_spot = len(coordinates)
expresion_value = # this is your actual expression list. It has length = total_spot
x_index = []
y_index = []
color = []
for i in range (0, total_spot):
    x_index.append(coordinates[i,0])
    y_index.append(coordinates[i,1])
    expresion_value_scaled = # expresion_value[i] is scaled between 0 to 1
    color.append((0,0,1,expresion_value_scaled))
    

for i in range (0, total_spot):  
    plt.scatter(x=x_index[i], y=y_index[i], color=color[j], s=10)   

save_path = '/cluster/project/schwartzlab/'
plt.savefig(save_path+'tissue_plot.svg', dpi=400)
plt.clf()
 

    
    
    
