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


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def get_colour_scheme(palette_name: str, num_colours: int) -> List[str]:
    """Extend a colour scheme using colour interpolation.

    Parameters
    ----------
    palette_name: The matplotlib colour scheme name that will be extended.
    num_colours: The number of colours in the output colour scheme.

    Returns
    -------
    New colour scheme containing 'num_colours' of colours. Each colour is a hex
    colour code.

    """
    scheme = [rgb2hex(c) for c in plt.get_cmap(palette_name).colors]
    if len(scheme) >= num_colours:
        return scheme[:num_colours]
    else:
        cmap = LinearSegmentedColormap.from_list("cmap", scheme)
        extended_scheme = cmap(np.linspace(0, 1, num_colours))
        return [to_hex(c, keep_alpha=False) for c in extended_scheme]
    
    
    
############
'''kmeans_label_file='/cluster/home/t116508uhn/64630/Tumur_64630_K-Means_7.csv'
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
      max = int(kmeans_label[i][1].split('_')[0].split(' ')[1])'''
############
#pathologist_label_file='/cluster/home/t116508uhn/64630/Annotation_KF.csv'
pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' # tumor_64630_D1_IX_annotation.csv'
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_label=dict()
count=np.zeros((4))
for i in range (1, len(pathologist_label)):
  if pathologist_label[i][1] == 'tumor': #'Tumour':
      barcode_label[pathologist_label[i][0]] = 1
      count[0] = count[0] + 1
  elif pathologist_label[i][1] == 'stroma_deserted': #'Stroma':
      barcode_label[pathologist_label[i][0]] = 2
      count[1] = count[1] + 1
  elif pathologist_label[i][1] == 'acinar_reactive': #'Acinar_reactive':  
      barcode_label[pathologist_label[i][0]] = 3
      count[2] = count[2] + 1
  elif pathologist_label[i][1] == 'Artifact':  
      barcode_label[pathologist_label[i][0]] = 4
      count[3] = count[3] + 1
        
max = 4
############

coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/barcodes.tsv'

#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_noPCA_QuantileTransform_wighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/'+'coordinates.npy')
#barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
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
plt.savefig(save_path+'pathologists_plot_new.png', dpi=400)
#plt.savefig(save_path+'kmeans_spaceranger_plot.png', dpi=400)
plt.clf()
# 413 == barcode not found, 443 = not labeled

############################################################################################################
import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import altair as alt
from vega_datasets import data
import pandas as pd



########
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



for i in range (0, len(colors)): 
    colors[i] = matplotlib.colors.to_hex([colors[i][0], colors[i][1], colors[i][2], colors[i][3]])

#####


#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new_noPCA/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label_node_embedding.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label_node_embedding.csv'


toomany_label_file='/cluster/home/t116508uhn/64630/TAGConv_test_r4_too-many-cell-clusters_org.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label.csv'
#toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label.csv'
#toomany_label_file='/cluster/home/t116508uhn/64630/GCN_r7_toomanycells_minsize20_labels.csv'
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
        barcode_label[toomany_label[i][0]] = int(toomany_label[i][1])
        cluster_dict[int(toomany_label[i][1])]=1
        '''if int(toomany_label[i][1]) == 61:  
            barcode_label[toomany_label[i][0]] = 60
        if int(toomany_label[i][1]) == 88:  
            barcode_label[toomany_label[i][0]] = 87
        if int(toomany_label[i][1]) == 47:  
            barcode_label[toomany_label[i][0]] = 46
        if int(toomany_label[i][1]) == 12:  
            barcode_label[toomany_label[i][0]] = 11
        if int(toomany_label[i][1]) == 15:  
            barcode_label[toomany_label[i][0]] = 14'''
        
############

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

####

index_array = dict()
for i in range (0, len(cluster_label)):
    index_array[cluster_label[i]] = i
    


data_list=dict()
data_list['cluster_label']=[]
data_list['X']=[]
data_list['Y']=[]

for i in range (0, len(barcode_info)):
    data_list['cluster_label'].append(barcode_info[i][3])
    data_list['X'].append(barcode_info[i][1])
    data_list['Y'].append(-barcode_info[i][2])
    



data_list_pd = pd.DataFrame(data_list)



#######


chart = alt.Chart(data_list_pd).mark_point(filled=True).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    #alt.Size('pop:Q'),
    color=alt.Color('cluster_label:N', scale=alt.Scale(range=colors))
).configure_legend(labelFontSize=6, symbolLimit=50)


save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.html')



