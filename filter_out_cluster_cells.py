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

rdef read_h5(f, i=0):
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
    toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv'     
    toomany_label=[]
    with open(toomany_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            toomany_label.append(line)

    barcode_label=dict()
    cluster_dict=dict()
    max=0
    for i in range (1, len(toomany_label)):
        if len(toomany_label[i])>0 :
            barcode_label[toomany_label[i][0]] = int(toomany_label[i][1])
            cluster_dict[int(toomany_label[i][1])]=1
            
    barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv' # 1406
    barcode_info=[]
    #barcode_info.append("")
    i=0
    with open(barcode_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            barcode_info.append([line[0],-1])
            i=i+1
            
    #cluster_dict[-1]=1
    cluster_label=list(cluster_dict.keys())

    count=0   
    for i in range (0, len(barcode_info)):
        if barcode_info[i][0] in barcode_label:
            barcode_info[i][1] = barcode_label[barcode_info[i][0]]
        else:
            count=count+1
    
    data_fold = args.data_path #+args.data_name+'/'
    print(data_fold)
    
    adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)
    gene_list_all=scipy.sparse.csr_matrix.toarray(adata_h5.X)  # row = cells x genes # (1406, 36601)
    gene_ids = list(adata_h5.var_names) # 36601
    #cell_genes = defaultdict(list)
    
    for cell_index in range (0, gene_list_all.shape[0]):
        gene_list_temp=[]
        for gene_index in range (0, gene_list_all.shape[1]):
            if gene_list_all[cell_index][gene_index]>0:
                gene_list_temp.append(gene_ids[gene_index])
                
        barcode_info[cell_index].append(gene_list_temp)
        
    
   target_cluster_id = 59 # BB
   gene_list=[]
   for i in range (0, len(barcode_info)):
       if barcode_info[i][1] == target_cluster_id:
           gene_list = gene_list + barcode_info[i][2]
        
   
   gene_list = set(gene_list)
    
   
           



        
    
