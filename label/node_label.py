
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
####################  get the whole training dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--min_cells', type=float, default=5, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
parser.add_argument( '--Dim_PCA', type=int, default=200, help='The output dimention of PCA')
parser.add_argument( '--data_path', type=str, default='dataset/', help='The path to dataset')
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data_noPCA_QuantileTransform/', help='The folder to store the generated data')
args = parser.parse_args() 

data_fold = args.data_path+args.data_name+'/'
generated_data_fold = args.generated_data_path + args.data_name+'/'
if not os.path.exists(generated_data_fold):
    os.makedirs(generated_data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file=args.data_name+'_filtered_feature_bc_matrix.h5')
print(adata_h5)

adata_h5.obs['in_tissue']
