print('package loading')
import numpy as np
import csv
import pickle
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
from random import choices
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
    csv_record = df.values.tolist()
    # from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score
    total_edge = len(csv_record)
    edge_index = []
    for i in range (0, len(csv_record)):
        edge_index.append(i)

    new_sets = []
    for i in range (0, num_new_set):
        temp_set = choices(edge_index, k=(2*total_edge)/3)
        new_sets.append(new_sets)

    # replace new_sets[i] with attention scores
    for i in range (0, num_new_set):
        for j in range(0, len(new_sets[i])):
            
    

    df = pd.DataFrame(csv_record_final) # output 4
    df.to_csv(args.output_path + args.model_name+'_CCC_list_confident.csv', index=False, header=False)
    
    ###########################################################################################################################################

