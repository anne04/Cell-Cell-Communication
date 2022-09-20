import pandas as pd
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

  
signature_file='/cluster/home/t116508uhn/64630/GeneList_KF_22Aug10.csv' # 1406
signature_info=defaultdict(list)
#barcode_info.append("")
with open(signature_file) as file:
    tsv_file = csv.reader(file, delimiter=",")
    for line in tsv_file:
        if (line[0].find('Basal') > -1) or (line[0].find('Classical') > -1) :
            signature_info[line[0]].append(line[1])

signature_info=dict(signature_info)
#############################

for signature in signature_info.keys():
    gene_list=signature_info[signature]
    print(signature+"\n")
    for gene in gene_list:
        print(" \\\""+ gene +"\\\",", end = ' ')


for signature in signature_info.keys():
    gene_list=signature_info[signature]
    print("\n"+signature+"\n")
    for gene in gene_list:
        print(" (\\\""+ gene +"\\\", Exact 0),", end = ' ')

        
        
        
        
 
