import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score


#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new_noPCA/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_label=[]
with open(barcode_file) as file:
    csv_file = csv.reader(file, delimiter="\t")
    for line in csv_file:
        barcode_label.append(line)
        
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label_node_embedding.csv'

#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv' #'/cluster/home/t116508uhn/64630/PCA_64embedding_Kena_label_l1mp5_temp.csv'
toomany_label_file='/cluster/home/t116508uhn/64630/GCN_r4_toomanycells_minsize20_labels.csv' #GCN_r4_toomanycells_org_labels.csv' # #GCN_r7_toomanycells_minsize20_labels.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/TAGConv_test_r4_too-many-cell-clusters.csv' #_org.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/spaceranger_pathologist.csv'
#toomany_label_file="/cluster/home/t116508uhn/64630/spaceranger_too-many-cells.csv"
toomany_label=[]
with open(toomany_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        toomany_label.append(line)

barcode_label_pred=dict()
cluster_dict=defaultdict(list)
max=0
for i in range (1, len(toomany_label)):
    if len(toomany_label[i])>0 :
        barcode_label_pred[toomany_label[i][0]] = int(toomany_label[i][1])
        cluster_dict[int(toomany_label[i][1])].append(toomany_label[i][0])

print(len(cluster_dict.keys()))
#################################################################################       
pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' # tumor_64630_D1_IX_annotation.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_label_pathologist=dict()
count=np.zeros((4))

'''for i in range (1, len(pathologist_label)):
  if pathologist_label[i][1] == 'tumor': #'Tumour':
      barcode_label_pathologist[pathologist_label[i][0]] = 1
  
  else:
      barcode_label_pathologist[pathologist_label[i][0]] = 0
'''

true_label_dict = defaultdict(list)
for i in range (1, len(pathologist_label)):
  
  if pathologist_label[i][1] == 'tumor': #'Tumour':
      barcode_label_pathologist[pathologist_label[i][0]] = 0
      true_label_dict[0].append(pathologist_label[i][0]) 
  elif pathologist_label[i][1] == 'stroma_deserted': #'Stroma':
      barcode_label_pathologist[pathologist_label[i][0]] = 1
      true_label_dict[1].append(pathologist_label[i][0]) 
  elif pathologist_label[i][1] == 'acinar_reactive': #'Acinar_reactive':  
      barcode_label_pathologist[pathologist_label[i][0]] = 2
      true_label_dict[2].append(pathologist_label[i][0]) 
  elif pathologist_label[i][1] == 'Artifact':  
      barcode_label_pathologist[pathologist_label[i][0]] = 3
      true_label_dict[3].append(pathologist_label[i][0]) 
      
      
      
      
#################################################################################  
kena_label_file='/cluster/home/t116508uhn/64630/Tumur_64630_K-Means_7.csv' #IX_annotation_artifacts.csv' # 
kena_label=[]
with open(kena_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        kena_label.append(line)

barcode_label_kena=dict()
count=np.zeros((4))
for i in range (1, len(kena_label)):
  cluster_id = int(kena_label[i][1].split('_')[0].split(' ')[1])
  barcode_label_kena[kena_label[i][0]] = cluster_id
     
      
 #################################################################################  

# node vs kena
spot_node_pred = []
spot_real = []
barcode_keys=list(barcode_label_pred.keys())
for barcode in barcode_keys:
    if barcode in barcode_label_kena:
        spot_real.append(barcode_label_kena[barcode])
        spot_node_pred.append(barcode_label_pred[barcode])
        

#print(normalized_mutual_info_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs kena: 0.4
print('homogeneity: kena: %g '%homogeneity_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs kena: .66

 #################################################################################  

# node vs path
spot_node_pred = []
spot_real = []
barcode_keys=list(barcode_label_pred.keys())
for barcode in barcode_keys:
    if barcode in barcode_label_pathologist:
        spot_real.append(barcode_label_pathologist[barcode])
        spot_node_pred.append(barcode_label_pred[barcode])
        

#print(normalized_mutual_info_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.10
print('homogeneity: pathologist: %g '%homogeneity_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.33
print('ARI: pathologist: %g '%adjusted_rand_score(labels_true=spot_real, labels_pred=spot_node_pred) )
    

#############################
cluster_list = np.zeros((len(cluster_dict.keys())))
true_label_list = np.zeros((len(true_label_dict.keys())))
    
max_matched = np.zeros((len(cluster_dict.keys())))
for k in range (0, len(cluster_dict.keys())):
    p = cluster_dict[k] 
    temp_max = 0
    for t in true_label_list.keys():  
        count = 0
        barcodes_list = true_label_list[t]
        for barcode in barcodes_list:
            if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
                if barcode in cluster_dict[p]:
                    count = count+1
                
        if count > temp_max: 
            temp_max = count
    
    max_matched[k] = temp_max
    
 N_cells = 0
 for barcode in barcode_label:
    if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
        N_cells = N_cells + 1
        
 purity_cluster = np.sum(max_matched)/N_cells 
        
        
        
