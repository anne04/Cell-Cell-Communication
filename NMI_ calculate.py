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
from sklearn.metrics.cluster import normalized_mutual_info_score


#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new_noPCA/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label_node_embedding.csv'

#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label.csv'
toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv' #'/cluster/home/t116508uhn/64630/PCA_64embedding_Kena_label_l1mp5_temp.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/spaceranger_pathologist.csv'
toomany_label=[]
with open(toomany_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        toomany_label.append(line)

barcode_label_pred=dict()
cluster_dict=dict()
max=0
for i in range (1, len(toomany_label)):
    if len(toomany_label[i])>0 :
        barcode_label_pred[toomany_label[i][0]] = int(toomany_label[i][1])

#################################################################################       
pathologist_label_file='/cluster/home/t116508uhn/64630/tumor_64630_D1_IX_annotation.csv' #IX_annotation_artifacts.csv' # 
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

for i in range (1, len(pathologist_label)):
  if pathologist_label[i][1] == 'tumor': #'Tumour':
      barcode_label_pathologist[pathologist_label[i][0]] = 1
      
  elif pathologist_label[i][1] == 'stroma_deserted': #'Stroma':
      barcode_label_pathologist[pathologist_label[i][0]] = 2
      
  elif pathologist_label[i][1] == 'acinar_reactive': #'Acinar_reactive':  
      barcode_label_pathologist[pathologist_label[i][0]] = 3
      
  elif pathologist_label[i][1] == 'Artifact':  
      barcode_label_pathologist[pathologist_label[i][0]] = 4
      
      
      
      
      
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
  '''
  if cluster_id == 2 or cluster_id == 4 or cluster_id == 7 : 
      barcode_label_kena[kena_label[i][0]] = 0
  else:
      barcode_label_kena[kena_label[i][0]] = 1
  '''      
      
 #################################################################################  

# node vs kena
spot_node_pred = []
spot_real = []
barcode_keys=list(barcode_label_pred.keys())
for barcode in barcode_keys:
    if barcode in barcode_label_kena:
        spot_real.append(barcode_label_kena[barcode])
        spot_node_pred.append(barcode_label_pred[barcode])
        

print(normalized_mutual_info_score(spot_real,spot_node_pred)) # 0.4

 #################################################################################  

# node vs path
spot_node_pred = []
spot_real = []
barcode_keys=list(barcode_label_pred.keys())
for barcode in barcode_keys:
    if barcode in barcode_label_pathologist:
        spot_real.append(barcode_label_pathologist[barcode])
        spot_node_pred.append(barcode_label_pred[barcode])
        

print(normalized_mutual_info_score(spot_real,spot_node_pred)) # 0.1

     
    

      
     
        
        
        
        
