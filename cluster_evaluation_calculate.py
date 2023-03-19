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
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score


barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_label=[]
with open(barcode_file) as file:
    csv_file = csv.reader(file, delimiter="\t")
    for line in csv_file:
        barcode_label.append(line[0])
        
#################################################################################               
toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv'
toomany_label=[]
with open(toomany_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        toomany_label.append(line)

barcode_label_pred=dict()
cluster_dict=defaultdict(list)

for i in range (1, len(toomany_label)):
    if len(toomany_label[i])>0 :
        barcode_label_pred[toomany_label[i][0]] = int(toomany_label[i][1])
        cluster_dict[int(toomany_label[i][1])].append(toomany_label[i][0])

print('total number of clusters in too-many-cells: %d '%len(cluster_dict.keys()))

#################################################################################       
pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' # tumor_64630_D1_IX_annotation.csv' #
pathologist_label=[]
cluster_dict=defaultdict(list)
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)
        line[1]
        
        
        
        
barcode_label_pathologist=dict()
count=np.zeros((4))

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
      
      
###################### Homogeneity and ARI ###########################################################  

# node vs path
spot_node_pred = []
spot_real = []
barcode_keys=list(barcode_label_pred.keys())
for barcode in barcode_keys:
    if barcode in barcode_label_pathologist:
        spot_real.append(barcode_label_pathologist[barcode])
        spot_node_pred.append(barcode_label_pred[barcode])
        

#print(normalized_mutual_info_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.10
print('Homogeneity: pathologist: %g '%homogeneity_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.33
print('ARI: pathologist: %g '%adjusted_rand_score(labels_true=spot_real, labels_pred=spot_node_pred) )
    

############### purity ##############

max_matched = np.zeros((len(cluster_dict.keys())))
cluster_list = list(cluster_dict.keys())

for k in range (0, len(cluster_list)):
    p = cluster_list[k]
    temp_max = 0
    for t in true_label_dict.keys():  
        count = 0
        barcodes_list = true_label_dict[t]
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
print('purity: pathologist: %g'%purity_cluster)       
        
##############  entropy ###############

entropy_cluster = np.zeros((len(cluster_dict.keys())))
cluster_list = list(cluster_dict.keys())


for k in range (0, len(cluster_list)):
    p = cluster_list[k]
    cluster_count = len(cluster_dict[p])
    
    H_sum = 0
    for t in true_label_dict.keys():  
        count_match = 0
        barcodes_list = true_label_dict[t]
        for barcode in barcodes_list:
            if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
                if barcode in cluster_dict[p]:
                    count_match = count_match+1
        
        #print(count_match/cluster_count)
        if count_match/cluster_count != 0:
            H_sum = H_sum + (count_match/cluster_count)*np.log(count_match/cluster_count)
    
    
    entropy_cluster[k] = H_sum
    
    
N_cells = 0
for barcode in barcode_label:
    if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
        N_cells = N_cells + 1

        
        
entropy_total = 0
for k in range (0, len(cluster_list)):
    entropy_total = entropy_total + (len(cluster_dict[cluster_list[k]])*entropy_cluster[k])/N_cells

entropy_total = - entropy_total
print('entropy_total: pathologist: %g'%entropy_total)           
