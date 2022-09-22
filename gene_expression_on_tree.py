s import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import scipy
import matplotlib
matplotlib.use('Agg')
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
import gseapy as gp
import csv
import stlearn as st
from collections import defaultdict
####################  get the whole training dataset

  
signature_file='/cluster/home/t116508uhn/64630/Geneset_22Sep21_Subtypesonly.csv' # 1406
signature_info=defaultdict(list)
#barcode_info.append("")
with open(signature_file) as file:
    tsv_file = csv.reader(file, delimiter=",")
    for line in tsv_file:
        if (line[0].find('Basal') > -1) or (line[0].find('Classical') > -1) :
            signature_info[line[0]].append(line[1])

signature_info=dict(signature_info)
#############################
signature_list = list(signature_info.keys())

for j in range (0, len(signature_list)):
    signature = signature_list[j]
    gene_list=signature_info[signature]
    print("\n"+signature)
    gene_list_str =""
    
    for i  in range (0, len(gene_list)-1):
        gene = gene_list[i]
        #print(" \\\""+ gene +"\\\",", end = ' ')
        gene_list_str = gene_list_str + " \\\""+ gene +"\\\","
        
    gene = gene_list[i+1]
    gene_list_str = gene_list_str + " \\\""+ gene +"\\\""
    command_str = "too-many-cells make-tree --matrix-path /cluster/home/t116508uhn/64630/spaceranger_output_new/ --labels-file /cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv --prior /cluster/home/t116508uhn/64630/TAGConv_test_r4_org/ --feature-column 2   --output /cluster/home/t116508uhn/64630/gene_expression_tree --draw-node-number --draw-leaf 'DrawItem (DrawContinuous ["+ gene_list_str +"])' --draw-scale-saturation 10  > /cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv"
    output_log=os.system(command_str)    
    
    
for signature in signature_info.keys():
    gene_list=signature_info[signature]
    print("\n"+signature+"\n")
    for gene in gene_list:
        print(" (\\\""+ gene +"\\\", Exact 0),", end = ' ')

        
        
        
        
 
