
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
      
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new_noPCA/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv' #'/cluster/home/t116508uhn/64630/PCA_64embedding_Kena_label_l1mp5_temp.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/spaceranger_pathologist.csv'
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
        if toomany_label[i][0] in barcode_tumor:
            barcode_label[toomany_label[i][0]] = int(toomany_label[i][1])
            cluster_dict[int(toomany_label[i][1])]=1
        else:
            barcode_label[toomany_label[i][0]] = -2
            cluster_dict[-2]=1
            
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],-1])
        i=i+1
cluster_dict[-1]=1
cluster_label=list(cluster_dict.keys())
count=0   
for i in range (0, len(barcode_info)):
    if barcode_info[i][0] in barcode_label:
        barcode_info[i][3] = barcode_label[barcode_info[i][0]]
    else:
        count=count+1
        
print(count)
print(len(cluster_label))

number = 20
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, number)]
number = 20
cmap = plt.get_cmap('tab20b')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]
colors=colors+colors_2
number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]
colors=colors+colors_2
number = 8
cmap = plt.get_cmap('Set2')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]
colors=colors+colors_2
number = 12
cmap = plt.get_cmap('Set3')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]
colors=colors+colors_2

cell_count_cluster=np.zeros((len(cluster_label)))
k=0
for j in range (0, len(cluster_label)):
    label_i = cluster_label[j]
    x_index=[]
    y_index=[]
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == label_i:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            cell_count_cluster[j] = cell_count_cluster[j]+1
            
    if label_i == -2 :
        set_color = '#808080'
    else:
        set_color = colors[k]
        k = k+1
        
    plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=set_color)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    
plt.legend(fontsize=5,loc='upper left')

save_path = '/cluster/home/t116508uhn/64630/'
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.png', dpi=400)
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.png', dpi=400)
plt.clf()
       
      
      
      
      
      
      
      
      
      
