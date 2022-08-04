import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group
from sklearn.preprocessing import quantile_transform
####################  get the whole training dataset
import scipy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--min_cells', type=float, default=5, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
parser.add_argument( '--Dim_PCA', type=int, default=200, help='The output dimention of PCA')
parser.add_argument( '--data_path', type=str, default='dataset/', help='The path to dataset')
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data_test/', help='The folder to store the generated data')
args = parser.parse_args()



print("hello world!main")
data_fold = args.data_path+args.data_name+'/'
print(data_fold)
generated_data_fold = args.generated_data_path + args.data_name+'/'
if not os.path.exists(generated_data_fold):
    os.makedirs(generated_data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
#count = adata_h5.X
#features = adata_preprocess(adata_h5, min_cells=args.min_cells, pca_n_comps=args.Dim_PCA)
i_adata=adata_h5
sc.pp.filter_genes(i_adata, min_cells=args.min_cells)
print(i_adata.shape)
adata_X = quantile_transform(scipy.sparse.csr_matrix.toarray(i_adata.X),copy=True)
adata_X = scipy.sparse.csr_matrix(adata_X)

np.save(generated_data_fold + 'features.npy',adata_X)

X_data = np.load(generated_data_fold + 'features.npy', allow_pickle=True)

import pickle
with open(generated_data_fold + 'features', 'wb') as fp:
    pickle.dump(adata_X, fp)
    
with open(generated_data_fold + 'features', 'rb') as fp:
    X_data = pickle.load(fp)
