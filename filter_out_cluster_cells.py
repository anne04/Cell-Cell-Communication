import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import scipy
import matplotlibmatplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group
import qnorm
#from sklearn.preprocessing import quantile_transform
import pickle
from scipy import sparse
import pickle
import scipy.linalg
from sklearn.metrics.pairwise import euclidean_distances
####################  get the whole training dataset


#rootPath = os.path.dirname(sys.path[0])
#os.chdir(rootPath+'/CCST')

print("hello world!")

def read_h5(f, i=0):
    print("hello world! read_h5")
    for k in f.keys():
        if isinstance(f[k], Group):
            print('Group', f[k])
            print('-'*(10-5*i))
            read_h5(f[k], i=i+1)
            print('-'*(10-5*i))
        elif isinstance(f[k], Dataset):
            print('Dataset', f[k])
            print(f[k][()])
        else:
            print('Name', f[k].name)
    print("hello world! read_h5_done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/', help='The path to dataset')
    parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
    args = parser.parse_args()

    main(args)
    

def main(args):
    print("hello world! main")
    data_fold = args.data_path #+args.data_name+'/'
    print(data_fold)
    
    adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)
    
    gene_list_all=scipy.sparse.csr_matrix.toarray(adata_h5.X)  # row = cells x genes # (1406, 36601)
    gene_ids = list(adata_h5.var_names) # 36601
    
    barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv' # 1406
    
    cell_genes = defaultdict(list)
    
    
    
    
    
