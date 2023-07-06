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
import copy 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
args = parser.parse_args()


print(args.data_path)
adata_h5 = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
####################################################
#temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))   # CHANGE: try doing  logCPM instead since singleR prefers that
#gene_vs_cell = temp
#np.save("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_quantile_transformed_"+args.data_name, gene_vs_cell)
####################################################
sc.pp.normalize_total(adata_h5, target_sum=1e6) # if target_sum=1e6, this is CPM normalization
print(adata_h5)
sc.pp.log1p(adata_h5)
gene_vs_cell = sparse.csr_matrix.toarray(adata_h5.X)
print('max gene expression value %g'%np.max(gene_vs_cell))
print('min gene expression value %g'%np.min(gene_vs_cell))
np.save("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_logCPM_"+args.data_name, gene_vs_cell)
###################################
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
cell_barcode = np.array(adata_h5.obs.index)
#############################################

df = pd.DataFrame(gene_ids)
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/gene_ids_'+args.data_name+'.csv', index=False, header=False)

df = pd.DataFrame(cell_barcode)
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cell_barcode_'+args.data_name+'.csv', index=False, header=False)
     
  
  
