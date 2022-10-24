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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



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
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'gat_r1_2attr_withfeature_onlyccc_bidir_97'+ '_attention.npy'
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 


attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[1][index][0]
    distribution.append(attention_scores[i][j])
    
    
'''for i in range (0, len(barcode_info)):
    if attention_scores[i][192]!=0:
        print('%d is %g'%(i, attention_scores[i][192]))'''
        
threshold =  np.percentile(sorted(distribution), 60)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
for i in range (0, attention_scores.shape[0]):
    for j in range (0, attention_scores.shape[1]):
        if attention_scores[i][j] > threshold: #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
             

############

graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph, directed=True, connection = 'weak', return_labels=True)
print('number of component %d'%n_components)

for i in range (0, len(barcode_info)):
#    if barcode_info[i][0] in barcode_label:
     barcode_info[i][3] = labels[i]
       

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



cell_count_cluster=np.zeros((labels.shape[0]))

for j in range (0, n_components):
    label_i = j
    x_index=[]
    y_index=[]
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == label_i:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            cell_count_cluster[j] = cell_count_cluster[j]+1
            spot_color = colors[j]
            ###############
            '''if barcode_info[i][3] == 61:  
                spot_color = colors[j-1]
            elif barcode_info[i][3] == 88:  
                spot_color = colors[j-1]
            elif barcode_info[i][3] == 47:  
                spot_color = colors[j-1]
            elif barcode_info[i][3] == 12:  
                spot_color = colors[j-1]'''
            #if barcode_info[i][3] == 15:  
            #    barcode_label[toomany_label[i][0]] = 14

            ###############
            
            
        
    plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=spot_color)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    
plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()


##############################


for i in range (0, len(colors)): 
    colors[i] = matplotlib.colors.to_hex([colors[i][0], colors[i][1], colors[i][2], colors[i][3]])

#####

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



