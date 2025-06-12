print('package loading')
import numpy as np
import csv
import pickle
import statistics
from scipy import sparse
from scipy import stats 
import scipy.io as sio
import scanpy as sc 
import matplotlib
matplotlib.use('Agg') 
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
#from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
import gzip
from kneed import KneeLocator
import copy 
import argparse
import gc
import os
import altair as alt
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")
import warnings
warnings.filterwarnings('ignore')
import anndata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir', help='The name of dataset') #, required=True) # default='',
    #_geneCorr_remFromDB
    #LRbind_GSM6177599_NYU_BRCA0_Vis_processed_1D_manualDB_geneCorr_bidir #LGALS1, PTPRC
    #LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir
    #LRbind_CID44971_1D_manualDB_geneCorr_bidir, CXCL10-CXCR3
    #LRbind_LUAD_1D_manualDB_geneCorr_signaling_bidir
    #'LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir
    #'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir'
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
    #######################################################################################################
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
    parser.add_argument( '--output_path', type=str, default='/cluster/home/t116508uhn/LRbind_output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--target_ligand', type=str, default='CCL19', help='') #
    parser.add_argument( '--target_receptor', type=str, default='CCR7', help='')
    args = parser.parse_args()
    
    args.output_path = args.output_path 
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

 
set_pre_files = ['LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_prefiltered/model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_prefiltered/model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR',\
                 'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_prefiltered/model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_prefiltered/model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR',
                 'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR'
                ]
set_post_files = ['LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir/model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir/model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR',\
                 'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir/model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir/model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR',\
                 'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L_down_up_deg_lr_list_sortedBy_totalScore_top_elbow_allLR',\
                 'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L_down_up_deg_novel_lr_list_sortedBy_totalScore_top_elbow_novelsOutOfallLR'
                 ]
flag = ['', '_novelOnly', '', '_novelOnly', '', '_novelOnly']
i = 3
set_pre_lrp = pd.read_csv(args.output_path +set_pre_files[i]+'.csv')
set_pre_lrp = list(set_pre_lrp['Ligand-Receptor Pairs'])

set_post_lrp = pd.read_csv(args.output_path + set_post_files[i]+'.csv')
set_post_lrp = list(set_post_lrp['Ligand-Receptor Pairs'])

common_lr = list(set(set_pre_lrp) & set(set_post_lrp))
print('Only pre %d, only post %d, common %d'%(len(set_pre_lrp)-len(common_lr), len(set_post_lrp)-len(common_lr), len(common_lr)))
pd.DataFrame(common_lr).to_csv(args.output_path +args.data_name+'_top_elbow'+'_common_lrp_pre_vs_post_filter'+flag[i]+'.csv', index=False)
