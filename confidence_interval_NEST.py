print('package loading')
import numpy as np
import csv
import pickle
import statistics
from scipy import sparse
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
from scipy.sparse.csgraph import connected_components
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
import gzip
#from kneed import KneeLocator
import copy 
import argparse
import gc
import os


##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--ccc_list_path', type=str, default='output/', help='CCC list path') # default='PDAC_64630',
    parser.add_argument( '--model_name', type=str, help='Name of the trained model', required=True)
    parser.add_argument( '--data_name', type=str, help='Name of the data.', required=True)
    parser.add_argument( '--output_path', type=str, default='output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    args = parser.parse_args()
    if args.output_path=='output/':
        args.output_path = args.output_path + args.data_name + '/'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.ccc_list_path=='output/':
        args.ccc_list_path = args.ccc_list_path + args.data_name + '/'
        
    ccc_list_filename = args.ccc_list_path + args.model_name+'_allCCC.csv'
##################### get metadata: barcode_info ###################################
    df = pd.read_csv(ccc_list_filename, sep=",")
    csv_record = [list(df.columns)] + df.values.tolist()
    # from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score
     


    df = pd.DataFrame(csv_record_final) # output 4
    df.to_csv(args.output_path + args.model_name+'_CCC_list_confident.csv', index=False, header=False)
    
    ###########################################################################################################################################

