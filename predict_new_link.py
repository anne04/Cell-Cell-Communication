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
 

X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'gat_r1_2attr_withfeature_onlyccc_97'+ '_attention.npy'
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
        
threshold =  np.percentile(sorted(distribution), 80)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))

for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] > threshold: #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_onlyccc_97', 'rb') as fp:
    row_col, edge_weight = pickle.load(fp)
    
input_edges = np.zeros((len(barcode_info),len(barcode_info)))  
for i in range (0, len(row_col)):
    input_edges[row_col[i][0]][row_col[i][1]]=edge_weight[i]
    

    
    
    
    
    
  
      
