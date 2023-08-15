import numpy as np
import csv
import pickle
from scipy import sparse
import scanpy as sc
import matplotlib
matplotlib.use('Agg') 
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
#from typing import List
import qnorm
from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd
import gzip
#import copy 
import argparse
import os

#sys.path.append("/home/gw/code/utility/altairThemes/")
#if True:  # In order to bypass isort when saving
#    import altairThemes
import altairThemes
import altair as alt
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")
##########################################################
# readCsv, preprocessDf, plot: these three functions are taken from GW's repository                                                                                                                                                                     /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/fatema_signaling/hist.py                                                                                                                                                                                         
import scipy.stats


def preprocessDf(df):
  """Transform ligand and receptor columns."""
  df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
  df["component"] = df["component"] #.astype(str).str.zfill(2)

  return df


def plot(df):
  set1 = altairThemes.get_colour_scheme("Set1", len(df["component"].unique()))
  set1[0] = '#000000'
  base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale = alt.Scale(range=set1)), 
            order=alt.Order("component:N", sort="ascending"),
            tooltip=["component"]
        )
  p = base

  return p
##alt.Color("component:N", scale = alt.Scale(range=set1))
####################### Set the name of the sample you want to visualize ###################################

data_name = 'PDAC_64630' #LUAD_GSM5702473_TD1
component_of_interest = [14, 15, 47, 60, 29, 38, 83, 17, 18] 
current_directory = '/cluster/home/t116508uhn/64630/'

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
args = parser.parse_args()
'''
elif data_name == 'PDAC_140694':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_C1_PDA_140694_Pa_P_Spatial10x/outs/' , help='The path to dataset') 
    parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
    parser.add_argument( '--data_name', type=str, default='PDAC_140694', help='The name of dataset')
    args = parser.parse_args()
'''
####### get the gene id, cell barcode, cell coordinates ######

adata_h5 = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
cell_barcode = np.array(adata_h5.obs.index)
temp = adata_h5.X

######################### get the cell vs gene matrix - we don not need that. Just in case someone is interested ##################
'''
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(temp)))  
cell_vs_gene = np.transpose(temp)  
'''
######################### get the cell vs cell distance matrix - we don not need that. Just in case someone is interested ##################
'''
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)
'''
##################### make cell metadata: barcode_info ###################################  
i=0
barcode_info=[]
for cell_code in cell_barcode:
    barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1], 0]) # last entry will hold the component number later
    i=i+1

####### load annotations ##############################################

pathologist_label_file='/cluster/home/t116508uhn/IX_annotation_artifacts.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)	
    
barcode_type=dict() # record the type (annotation) of each spot (barcode)
for i in range (1, len(pathologist_label)):
    barcode_type[pathologist_label[i][0]] = pathologist_label[i][1]
    

######################### read the NEST output in csv format ####################################################

# # columns are: from_cell, to_cell, ligand_gene, receptor_gene, attention_score, component, from_id, to_id
filename_str = 'NEST_combined_output_'+args.data_name+'.csv'
inFile = current_directory +filename_str 
df = pd.read_csv(inFile, sep=",")
csv_record_final = df.values.tolist()
df_column_names = list(df.columns)
csv_record_final = [df_column_names] + csv_record_final

######################## count how many unique components ###################################### 
component_list = dict()
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    record = csv_record_final[record_idx]
    component_label = record[5]
    component_list[component_label] = ''
    
component_list[0] = ''
unique_component_count = len(component_list.keys())

######################## filter the records ########################################################

## change the csv_record_final here if you want histogram for specific components/regions only. e.g., if you want to plot only stroma region, or tumor-stroma regions etc.    ##

temp_csv_record_final = []
temp_csv_record_final.append(csv_record_final[0]) #last entry is a dummy for histograms

for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    # if at least one of ligand and receptors are tumors: 
    if (barcode_type[csv_record_final[record_idx][0]] == 'tumor' or barcode_type[csv_record_final[record_idx][1]] == 'tumor'): # or (barcode_type[csv_record_final[record_idx][0]] != 'tumor' and barcode_type[csv_record_final[record_idx][1]] != 'tumor')):
        if csv_record_final[record_idx][5] in component_of_interest:
            temp_csv_record_final.append(csv_record_final[record_idx]) # it is considered during making histogram. 
	
temp_csv_record_final.append(csv_record_final[len(csv_record_final)-1]) #last entry is a dummy for histograms
csv_record_final = temp_csv_record_final
###################################  Histogram plotting #################################################################################
'''
set1 = altairThemes.get_colour_scheme("Set1", unique_component_count)
set1[0] = '#000000'
csv_record_final[0].append('comp_color')
for record_idx in range (1, len(csv_record_final)): 
    csv_record_final[record_idx].append(set1[int(csv_record_final[record_idx][5])])
'''

	
df = pd.DataFrame(csv_record_final)
df.to_csv(current_directory+'temp_csv.csv', index=False, header=False)
df = pd.read_csv(current_directory+'temp_csv.csv', sep=",")
os.remove(current_directory+'temp_csv.csv') # delete the intermediate file
df = preprocessDf(df)
p = plot(df) #, unique_component_count
outPath = current_directory+'histogram_test.html'
p.save(outPath)	

