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
from gseapy import gseaplot
import csv
import stlearn as st
from collections import defaultdict
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

#if __name__ == "__main__":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
args = parser.parse_args()

#    main(args)
    

def main(args):
    print("hello world! main")  
    toomany_label_file='/cluster/home/t116508uhn/64630/GCN_r4_toomanycells_minsize20_labels.csv'
    #toomany_label_file='/cluster/home/t116508uhn/64630/TAGConv_test_r4_too-many-cell-clusters_org.csv' #'/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv'#'/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv'     
    toomany_label=[]
    with open(toomany_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            toomany_label.append(line)

    barcode_label=dict()
    cluster_dict=dict()
    
    for i in range (1, len(toomany_label)):
        if len(toomany_label[i])>0 :
            barcode_label[toomany_label[i][0]] = int(toomany_label[i][1])
            if int(toomany_label[i][1]) in cluster_dict:
                cluster_dict[int(toomany_label[i][1])]=cluster_dict[int(toomany_label[i][1])]+1
            else:
                cluster_dict[int(toomany_label[i][1])]=1
    
    print(len(cluster_dict.keys()))
    print(cluster_dict)
    
    pathologist_label_file='/cluster/home/t116508uhn/64630/tumor_64630_D1_IX_annotation.csv' #IX_annotation_artifacts.csv' #
    pathologist_label=[]
    with open(pathologist_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            pathologist_label.append(line)

    barcode_tumor=dict()
    for i in range (1, len(pathologist_label)):
      if pathologist_label[i][1] == 'tumor': #'Tumour':
          barcode_tumor[pathologist_label[i][0]] = 1           
            
    barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv' # 1406
    barcode_info=[]
    #barcode_info.append("")
    i=0
    with open(barcode_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            barcode_info.append([line[0],-1,[]])
            i=i+1
            
    #cluster_dict[-1]=1
    cluster_label=list(cluster_dict.keys())

    count=0   
    for i in range (0, len(barcode_info)):
        if (barcode_info[i][0] in barcode_label) and (barcode_info[i][0] in barcode_tumor):
            barcode_info[i][1] = barcode_label[barcode_info[i][0]]
        else:
            count=count+1
    
    data_fold = args.data_path #+args.data_name+'/'
    print(data_fold)
    
    adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)
    sc.pp.filter_genes(adata_h5, min_cells=1)
    print(adata_h5)
    ####################   
    gene_ids = list(adata_h5.var_names)
    temp = qnorm.quantile_normalize(np.transpose(scipy.sparse.csr_matrix.toarray(adata_h5.X)))  #quantile_transform(scipy.sparse.csr_matrix.toarray(adata_h5.X), copy=True)
    adata_X = np.transpose(temp)    
    gene_list_all = adata_X
    #####################
    #sc.pp.highly_variable_genes(adata_h5,flavor='seurat_v3', n_top_genes = 500)
    #print(adata_h5)
    #adata_X = 
    #sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True)
    #print(adata_h5)
    #adata_X = 
    #sc.pp.scale(adata_h5)
    #print(adata_h5)
    
    #gene_list_all=adata_h5.X
    #gene_list_all=scipy.sparse.csr_matrix.toarray(adata_h5.X)  # row = cells x genes # (1406, 36601)
    #gene_ids = list(adata_h5.var_names) # 36601
    #cell_genes = defaultdict(list)
    
    for cell_index in range (0, gene_list_all.shape[0]):
        gene_list_temp=defaultdict(list)
        non_zero_index=list(np.where(gene_list_all[cell_index]!=0)[0])
        for gene_index in non_zero_index:
            if gene_list_all[cell_index][gene_index]>0:
                gene_list_temp[gene_ids[gene_index]].append(gene_list_all[cell_index][gene_index])
                
        barcode_info[cell_index][2]=gene_list_temp
    
    
    
    signature_file='/cluster/home/t116508uhn/64630/GeneList_KF_22Aug10.csv' # 1406
    signature_info=defaultdict(list)
    #barcode_info.append("")
    with open(signature_file) as file:
        tsv_file = csv.reader(file, delimiter=",")
        for line in tsv_file:
            if (line[0].find('Basal') > -1) or (line[0].find('Classical') > -1) :
                signature_info[line[0]].append(line[1])
    
    signature_info=dict(signature_info)

    target_cluster_id = [[25], [19], [69, 70, 72, 73], [52, 51], [37]]
    #target_cluster_id =[[60,61], [11,12], [14,15], [88,87], [46,47]] #[[61]] #[[11,12,15],[14]] #
    for target_cluster in target_cluster_id:
        print("cluster ID: ", target_cluster)
        gene_list_cluster=defaultdict(list)
        for i in range (0, len(barcode_info)):
            if barcode_info[i][1] in target_cluster:
                #print(barcode_info[i][1])
                for genes in list(barcode_info[i][2].keys()):
                    gene_list_cluster[genes] = gene_list_cluster[genes] + barcode_info[i][2][genes]

        for genes in list(gene_list_cluster.keys()):
            gene_list_cluster[genes]=[np.mean(gene_list_cluster[genes])]

        data_rnk=pd.DataFrame.from_dict(gene_list_cluster, orient='index')
 

        pre_res = gp.prerank(rnk = data_rnk,
                             gene_sets = signature_info,
                             threads=4,
                             min_size=5,
                             max_size=1000,
                             permutation_num=1000, # reduce number to speed up testing
                             outdir=None, # don't write to disk
                             seed=6,
                             #verbose=True, # see what's going on behind the scenes
                            )
        print(pre_res.res2d)
        name_str=str(target_cluster[0])
        if len(target_cluster)>1:
            for i in range (1, len(target_cluster)):
                name_str=name_str+"_"+str(target_cluster[i])
       
        terms = pre_res.res2d.Term
        for i in range (0, 6):
            # save figure
            # gseaplot(rank_metric=pre_res.ranking, term=terms[i], ofname=save_path+name_str+'_'+str(i)+'_prerank_tagconv_test_r4.svg', **pre_res.results[terms[i]])
            gseaplot(rank_metric=pre_res.ranking, term=terms[i], ofname=save_path+name_str+'_'+str(i)+'_prerank_GCN_r4.svg', **pre_res.results[terms[i]])


      



        
    
