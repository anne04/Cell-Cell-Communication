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


############
kmeans_label_file='/cluster/home/t116508uhn/64630/Tumur_64630_K-Means_7.csv'
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
############
pathologist_label_file='/cluster/home/t116508uhn/64630/tumor_64630_D1_IX_annotation.csv'
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_label=dict()
for i in range (1, len(pathologist_label)):
  if pathologist_label[i][1] == 'tumor':
      barcode_label[pathologist_label[i][0]] = 1
  elif pathologist_label[i][1] == 'stroma_deserted':
      barcode_label[pathologist_label[i][0]] = 2
  elif pathologist_label[i][1] == 'acinar_reactive':  
      barcode_label[pathologist_label[i][0]] = 3
        
max = 3
############


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
            
                
    plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = label_i, s=5)
    
print(count)
plt.legend()

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'pathologists_plot.png', dpi=400)
#plt.savefig(save_path+'kmeans_spaceranger_plot.png', dpi=400)
plt.clf()
# 413 == barcode not found, 443 = not labeled

############################################################################################################
toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_r3.csv'
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
############
cluster_label=list(cluster_dict.keys())
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
count=0   
for i in range (0, len(barcode_info)):
    if barcode_info[i][0] in barcode_label:
        barcode_info[i][3] = barcode_label[barcode_info[i][0]]
    else:
        count=count+1
        
print(count)

NUM_COLORS=len(cluster_label)
cm = plt.get_cmap('gist_rainbow')
colors=[cm(1.*i/(NUM_COLORS*500)) for i in range(NUM_COLORS*500)]

number = 20
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

number = 20
cmap = plt.get_cmap('tab20b')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

'''fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle('color', [cm(1.*i/(NUM_COLORS*5)) for i in range(NUM_COLORS*5)])
     
number = NUM_COLORS
cmap = plt.get_cmap('gnuplot')

colors=[]
colors.append([.1,.1,.1])    

start=len(colors)
for i in range (1, 10):
    colors.append([.1,.5,colors[0][2]+0.05*i*2])
    colors.append([colors[0][0]+0.05*i*2, 1, .3])
    colors.append([.5,colors[0][1]+0.05*i*2,.8])
    colors.append([colors[0][0]+0.05*i*2, .7, 0.5])'''

for j in range (0, len(cluster_label)):
    label_i=cluster_label[j]
    x_index=[]
    y_index=[]
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == label_i:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
    plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=colors[j], s=5)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10, s=5)
    
plt.legend()

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_r3_plot.png', dpi=400)
plt.clf()