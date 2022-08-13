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

kmeans_label_file='64630/Tumur_64630_K-Means_7.csv'
kmeans_label=[]
with open(kmeans_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        kmeans_label.append(line)

barcode_label=dict()
max=0
for i in range (1, len(kmeans_label)):
  barcode_label[kmeans_label[i][0]] = int(kmeans_label[i][1].split('_')[0].split(' ')[1])
  if max < int(kmeans_label[i][1].split('_')[0].split(' ')[1]):
      max = int(kmeans_label[i][1].split('_')[0].split(' ')[1])

coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_noPCA_QuantileTransform_wighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
        
for i in range (0, len(barcode_info)):
    if barcode_info[i][0] in barcode_label:
        barcode_info[i][3] = barcode_label[barcode_info[i][0]]
        
count=0      
for label_i in range (0, max+1):
    x_index=[]
    y_index=[]
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == label_i:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            if label_i==0:
                count=count+1
            
                
    plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = label_i)
    
print(count)
plt.legend()

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'kmeans_spaceranger_plot.png', dpi=400)
plt.clf()
# 413 == barcode not found, 443 = not labeled
