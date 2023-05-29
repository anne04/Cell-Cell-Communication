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


import os
import gc
import ot
import pickle
import anndata
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import commot as ct

threshold_distance = 2 #2 = path equally spaced
k_nn = 10 # #5 = h
distance_measure = 'knn'  #'threshold_dist' # <-----------
datatype = 'path_mixture_of_distribution' #'path_equally_spaced' #

options = 'dt-path_mixture_of_distribution_lrc8_cp100_noise0_random_overlap_knn_cellCount2534_f_3dim'

gene_vs_cell = pd.read_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_f.csv', index_col=0)  
cell_vs_gene = gene_vs_cell.transpose()
adata_synthetic = anndata.AnnData(cell_vs_gene)

df_x=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_x.csv',header=None)
df_y=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_y.csv',header=None)

gene_vs_cell = pd.read_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_f.csv', index_col=0)  



    
    

