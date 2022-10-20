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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
args = parser.parse_args()


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
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

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
        
############
X_attention_filename = args.embedding_data_path + args.data_name + '/' + args.model_name + '_attention.npy'
X_attention_bundle = np.load(X_attention_filename) 

attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[1][index][0]




############

        
for i in range (0, len(barcode_info)):
    if barcode_info[i][0] in barcode_label:
        barcode_info[i][3] = barcode_label[barcode_info[i][0]]
        

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



