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
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd
import gzip
from kneed import KneeLocator
import copy 
import altairThemes
import altair as alt
import argparse

spot_diameter = 89.43 #pixels
##########################################################
# written by GW                                                                                                                                                                     /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/fatema_signaling/hist.py                                                                                                                                                                                         
import scipy.stats

#sys.path.append("/home/gw/code/utility/altairThemes/")
#if True:  # In order to bypass isort when saving
#    import altairThemes

def readCsv(x):
  """Parse file."""
  #colNames = ["method", "benchmark", "start", "end", "time", "memory"]
  df = pd.read_csv(x, sep=",")

  return df

def preprocessDf(df):
  """Transform ligand and receptor columns."""
  df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
  df["component"] = df["component"] #.astype(str).str.zfill(2)

  return df

def statOrNan(xs, ys):
  if len(xs) == 0 or len(ys) == 0:
    return None
  else:
    return scipy.stats.mannwhitneyu(xs, ys)

def summarizeStats(df, feature):
  meanRes = df.groupby(["benchmark", "method"])[feature].mean()
  statRes = df.groupby("benchmark").apply(lambda x: post.posthoc_ttest(x, val_col = feature, group_col = "method", p_adjust = "fdr_bh"))

  return (meanRes, statRes)

def writeStats(stats, feature, outStatsPath):
  stats[0].to_csv(outStatsPath + "_feature_" + feature + "_mean.csv")
  stats[1].to_csv(outStatsPath + "_feature_" + feature + "_test.csv")

  return

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
'''
def plot(df):
  number = 20
  cmap = plt.get_cmap('tab20')
  colors = [cmap(i) for i in np.linspace(0, 1, number)]
  for i in range (0, len(colors)): 
    colors[i] = matplotlib.colors.to_hex([colors[i][0], colors[i][1], colors[i][2], colors[i][3]])
  
  #set1 = altairThemes.get_colour_scheme("Set1", len(df["component"].unique()))
  #set1[0] = '#000000'
  base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale = alt.Scale(range=colors)),
            order=alt.Order("component:N", sort="ascending"),
            tooltip=["component"]
        )
  p = base

  return p
'''
def totalPlot(df, features, outPath):

  p = alt.hconcat(*map(lambda x: plot(df, x), features))

  outPath = outPath + "_boxplot.html"

  p.save(outPath)

  return
##########################################################

data_name = 'PDAC_64630' #LUAD_GSM5702473_TD1

##########################################################
if data_name == 'LUAD_GSM5702473_TD1':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/LUAD/LUAD_GSM5702473_TD1/' , help='The path to dataset') 
    parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
    parser.add_argument( '--data_name', type=str, default='LUAD_GSM5702473_TD1', help='The name of dataset')
    parser.add_argument( '--model_name', type=str, default='gat_r1_3attr', help='model name')
    #parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
    args = parser.parse_args()  	
#############################################################   
elif data_name == 'V1_Human_Lymph_Node_spatial':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/' , help='The path to dataset') 
    parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
    parser.add_argument( '--data_name', type=str, default='V1_Human_Lymph_Node_spatial', help='The name of dataset')
    parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
    parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
    args = parser.parse_args()


elif data_name == 'PDAC_64630':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
    parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
    parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
    parser.add_argument( '--model_name', type=str, default='gat_2attr', help='model name')
    parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
    args = parser.parse_args()


elif data_name == 'PDAC_140694':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/V10M25-60_C1_PDA_140694_Pa_P_Spatial10x/outs/' , help='The path to dataset') 
    parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
    parser.add_argument( '--data_name', type=str, default='PDAC_140694', help='The name of dataset')
    #parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
    #parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
    args = parser.parse_args()

####### get the gene id, cell barcode, cell coordinates ######

if data_name == 'LUAD_GSM5702473_TD1':
    # read the mtx file
    temp = sc.read_10x_mtx(args.data_path)
    print(temp)
    #sc.pp.log1p(temp)
    sc.pp.filter_genes(temp, min_cells=1)
    print(temp)
    #sc.pp.highly_variable_genes(temp) #3952
    #temp = temp[:, temp.var['highly_variable']]
    #print(temp)
    
    gene_ids = list(temp.var_names) 
    cell_barcode = np.array(temp.obs.index)
    
    # now read the tissue position file. It has the format: 
    #df = pd.read_csv('/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/spatial/tissue_positions_list.csv', sep=",",header=None)   # read dummy .tsv file into memory
    df = pd.read_csv('/cluster/projects/schwartzgroup/fatema/data/LUAD/LUAD_GSM5702473_TD1/GSM5702473_TD1_tissue_positions_list.csv', sep=",",header=None)   # read dummy .tsv file into memory
    tissue_position = df.values
    barcode_vs_xy = dict() # record the x and y coord for each spot
    for i in range (0, tissue_position.shape[0]):
        barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][5], tissue_position[i][4]] #for some weird reason, in the .h5 format, the x and y are swapped
        #barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][4], tissue_position[i][5]] 
    
    coordinates = np.zeros((cell_barcode.shape[0], 2)) # insert the coordinates in the order of cell_barcodes
    for i in range (0, cell_barcode.shape[0]):
        coordinates[i,0] = barcode_vs_xy[cell_barcode[i]][0]
        coordinates[i,1] = barcode_vs_xy[cell_barcode[i]][1]
    

else:
    adata_h5 = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print(adata_h5)
    sc.pp.filter_genes(adata_h5, min_cells=1)
    print(adata_h5)
    gene_ids = list(adata_h5.var_names)
    coordinates = adata_h5.obsm['spatial']
    cell_barcode = np.array(adata_h5.obs.index)
    

##################### make cell metadata: barcode_info ###################################
 
i=0
barcode_serial = dict()
for cell_code in cell_barcode:
    barcode_serial[cell_code]=i
    i=i+1
    
i=0
barcode_info=[]
for cell_code in cell_barcode:
    barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1], 0]) # last entry will hold the component number later
    i=i+1
	
i=0
node_id_sorted_xy=[]
for cell_code in cell_barcode:
    node_id_sorted_xy.append([i, coordinates[i,0],coordinates[i,1]])
    i=i+1
	
node_id_sorted_xy = sorted(node_id_sorted_xy, key = lambda x: (x[1], x[2]))



####### load annotations ##############################################

'''
pathologist_label_file='/cluster/home/t116508uhn/human_lymphnode_Spatial10X_manual_annotations.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if pathologist_label[i][1] == 'GC': 
        barcode_type[pathologist_label[i][0]] = 1 
    #elif pathologist_label[i][1] =='':
    #    barcode_type[pathologist_label[i][0]] = 0 #'stroma_deserted'
    else:
        barcode_type[pathologist_label[i][0]] = 0
'''

'''
pathologist_label_file='/cluster/home/t116508uhn/64630/spot_vs_type_dataframe_V1_HumanLympNode.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)
	
spot_label = []
for i in range (1, len(pathologist_label)):
    spot_label.append([pathologist_label[i][0], float(pathologist_label[i][1]), float(pathologist_label[i][2]), float(pathologist_label[i][3])])
    
spot_label = sorted(spot_label, key = lambda x: x[3], reverse=True) # descending order of 
barcode_Tcell = []
barcode_B = []
barcode_GC = []
for i in range (0, len(spot_label)):
    if spot_label[i][1] >= (spot_label[i][2] + spot_label[i][3])*2:
        barcode_Tcell.append(spot_label[i][0])
    elif spot_label[i][2] >= (spot_label[i][1] + spot_label[i][3])*2:
        barcode_B.append(spot_label[i][0])
    elif spot_label[i][3] >= (spot_label[i][1] + spot_label[i][2])*2:
        barcode_GC.append(spot_label[i][0])
       
barcode_type=dict()
for i in range (0, len(barcode_Tcell)):
    barcode_type[barcode_Tcell[i]] = 1 # tcell
    
for i in range (0, len(barcode_B)):
    barcode_type[barcode_B[i]] = 2
  
for i in range (0, len(barcode_GC)):
    barcode_type[barcode_GC[i]] = 3
    
for i in range (0, len(spot_label)):
    if spot_label[i][0] not in barcode_type:
        barcode_type[spot_label[i][0]] = 0
#############################################################################
data_list=dict()
data_list['pathology_label']=[]
data_list['X']=[]
data_list['Y']=[]

for i in range (0, len(barcode_info)):
    data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
    data_list['X'].append(barcode_info[i][1])
    data_list['Y'].append(barcode_info[i][2])


data_list_pd = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", 4)
set1[0] = '#000000'

chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    shape = alt.Shape('pathology_label:N'), #shape = "pathology_label",
    color=alt.Color('pathology_label:N', scale=alt.Scale(range=set1)),
    tooltip=['pathology_label']
)

save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'V1_humanLymphNode.html') #   
        
'''

'''
########## sabrina ###########################################	
pathologist_label_file='/cluster/projects/schwartzgroup/fatema/find_ccc/singleR_spot_annotation_Sabrina.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if pathologist_label[i][1] == 'tumor': #'Tumour':
        barcode_type[pathologist_label[i][0]] = 1 #'tumor'
    elif pathologist_label[i][1] =='stroma_deserted':
        barcode_type[pathologist_label[i][0]] = 0 #'stroma_deserted'
    elif pathologist_label[i][1] =='acinar_reactive':
        barcode_type[pathologist_label[i][0]] = 2 #'acinar_reactive'
    else:
        barcode_type[pathologist_label[i][0]] = 0 #'zero' 
'''	
#################################################################
pathologist_label_file='/cluster/home/t116508uhn/IX_annotation_artifacts.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)	
	
barcode_type=dict()
for i in range (1, len(pathologist_label)):
    barcode_type[pathologist_label[i][0]] = pathologist_label[i][1]
    
		

'''
spot_type = []
pathologist_label_file='/cluster/home/t116508uhn/V10M25-060_C1_T_140694_Histology_annotation_IX.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)
        spot_type.append(line[1])
        
barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if 'tumor_LVI' in pathologist_label[i][1]:
        barcode_type[pathologist_label[i][0]] = 'tumor_LVI'
    elif 'tumor_PNI' in pathologist_label[i][1]:
        barcode_type[pathologist_label[i][0]] = 'tumor_PNI'
    elif 'tumor_stroma' in pathologist_label[i][1]:
        barcode_type[pathologist_label[i][0]] = 'tumor_stroma'
    elif 'tumor_vs_acinar' in pathologist_label[i][1]:
        barcode_type[pathologist_label[i][0]] = 'tumor_vs_acinar'     
    elif 'nerve' in pathologist_label[i][1]: 
        barcode_type[pathologist_label[i][0]] = 'nerve'
    elif 'vessel' in pathologist_label[i][1]: 
        barcode_type[pathologist_label[i][0]] = 'vessel'
    #elif pathologist_label[i][1] =='stroma_deserted':
    #    barcode_type[pathologist_label[i][0]] = 'stroma_deserted'
    #elif pathologist_label[i][1] =='acinar_reactive':
    #    barcode_type[pathologist_label[i][0]] = 'acinar_reactive'
    else:
        barcode_type[pathologist_label[i][0]] = 'others'
'''
#############################################################################
'''
ccc_too_many_cells_LUAD = pd.read_csv('/cluster/projects/schwartzgroup/fatema/CCST/exp2_D1_ccc_toomanycells_cluster.csv')
ccc_too_many_cells_LUAD_dict = dict()
for i in range(0, len(ccc_too_many_cells_LUAD)):
    ccc_too_many_cells_LUAD_dict[ccc_too_many_cells_LUAD['cell'][i]] = int(ccc_too_many_cells_LUAD['cluster'][i])

for i in range(0, len(barcode_info)):
    barcode_info[i][3] = ccc_too_many_cells_LUAD_dict[barcode_info[i][0]]
	

barcode_type=dict()
for i in range (0, len(barcode_info)):
    barcode_type[barcode_info[i][0]] = 0 
'''
##### load input graph ##########################

datapoint_size = len(barcode_info)
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+'all_avg', 'rb') as fp:  #b, a:[0:5]           
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
#    row_col, edge_weight = pickle.load(fp)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_3d', 'rb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_kneepoint_woBlankedge', 'rb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_omniPath_separate_'+'threshold_distance_density_kneepoint', 'rb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+ '_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_3d', 'rb') as fp: 
    row_col, edge_weight, lig_rec = pickle.load(fp) # density_

lig_rec_dict = []
for i in range (0, datapoint_size):
    lig_rec_dict.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []

total_type = np.zeros((2))        
for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        lig_rec_dict[i][j].append(lig_rec[index])  
############################################################################
'''
attention_scores = []
datapoint_size = len(barcode_info)
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []

distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    #if i==j:
    #    if len(lig_rec_dict[i][j])==0:
    #        continue 
    #if barcode_type[cell_barcode[i]]==1 and barcode_type[cell_barcode[j]]==1: #i in spot_interest_list and j in spot_interest_list:
    #if lig_rec[index][0]!='CCL19' or lig_rec[index][1] != "CCR7": #lig_rec[index][0]=='IL21' and lig_rec[index][1] == "IL21R": #
    #    continue
    if edge_weight[index][1]>0:
        attention_scores[i][j].append(edge_weight[index][1]) # * edge_weight[index][0]) # * edge_weight[index][2])
        distribution.append(edge_weight[index][1]) # * edge_weight[index][0]) # * edge_weight[index][2])
        ccc_index_dict[i] = ''
        ccc_index_dict[j] = ''   
        #lig_rec_dict[i][j].append(lig_rec[index])  
        
'''        
'''''' 

############# load output graph #################################################
filename = ["r1_", "r2_", "r3_", "r4_", "r5_", "r6_", "r7_", "r8_", "r9_", "r10_"]
total_runs = 5
start_index = 0
csv_record_dict = defaultdict(list)
for run_time in range (start_index, start_index+total_runs):
    gc.collect()
    #run_time = 2
    run = run_time
    print('run %d'%run)
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'merfish_data_Female_Virgin_ParentingExcitatory_id24_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_xyz_r1_th500'+filename[run_time]+'attention_l1.npy'   
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_140694_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_'+filename[run_time]+'attention_l1.npy'
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + args.data_name + '_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_'+filename[run_time]+'attention_l1.npy'
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnCCC_c_70_attention.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnCCC_c_all_avg_bothlayer_attention_l1.npy' #a
    X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_'+filename[run_time]+'attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_bothAbove_cell98th_3dim_'+filename[run_time]+'attention_l1.npy'
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_bothAbove_bothAbove_cell98th_'+filename[run_time]+'attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_withlrFeature_bothAbove_cell98th_'+filename[run_time]+'attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_bothAboveDensity_r2_attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_omnipath_threshold_distance_bothAboveDensity_attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_eitherAbove_cell_knee_'+filename[run_time]+'attention_l1.npy' #a
    #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'PDAC_cellchat_nichenet_threshold_distance_bothAbove_cell98th_scaled_'+filename[run_time]+'attention_l1.npy' #a

    X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) #_withFeature

    for l in [2, 3]: # = 3 # 3 = layer 1, 2 = layer 2
    
	    distribution = []
	    for index in range (0, X_attention_bundle[0].shape[1]):
	        i = X_attention_bundle[0][0][index]
	        j = X_attention_bundle[0][1][index]
	        #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
	        #    continue
	        distribution.append(X_attention_bundle[l][index][0])
	    
	    attention_scores = []
	    for i in range (0, datapoint_size):
	        attention_scores.append([])   
	        for j in range (0, datapoint_size):	
	            attention_scores[i].append([])   
	            attention_scores[i][j] = []
	            
	    min_attention_score = 1000
	    max_value = np.max(distribution)
	    min_value = np.min(distribution)
	    distribution = []
	    for index in range (0, X_attention_bundle[0].shape[1]):
	        i = X_attention_bundle[0][0][index]
	        j = X_attention_bundle[0][1][index]
	        #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
	        #    continue
	        scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
	        attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
	        if min_attention_score > scaled_score:
	            min_attention_score = scaled_score
	        distribution.append(scaled_score)
	        
	        
	    if min_attention_score<0:
	        min_attention_score = -min_attention_score
	    else: 
	        min_attention_score = 0
	    
	    print('min attention score %g, total edges %d'%(min_attention_score, len(distribution)))
	    ##############
	    #plt.clf()
	    #plt.hist(distribution, color = 'blue',bins = int(len(distribution)/5))
	    #save_path = '/cluster/home/t116508uhn/64630/'
	    #plt.savefig(save_path+'distribution_region_of_interest_'+filename[run_time]+'_l2attention_score.svg', dpi=400) # _CCL19_CCR7
	    #plt.savefig(save_path+'dist_bothAbove98th_3dim_'+filename[run_time]+'attention_score.svg', dpi=400) # output 1
	    #plt.savefig(save_path+'PDAC_140694_dist_bothAbove98th_3dim_tanh_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.savefig(save_path+'dist_'+args.data_name+'_bothAbove98th_3dim_tanh_h512_l2attention_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.savefig(save_path+'dist_'+args.data_name+'_bothAbove98th_3dim_tanh_h512_filtered_l2attention_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.savefig(save_path+'dist_bothAbove98th_wfeature_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.savefig(save_path+'dist_bothAbove98th_scaled_wfeature_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.savefig(save_path+'dist_bothAbove98th_'+filename[run_time]+'attention_score.svg', dpi=400)
	    #plt.clf()
	    ##############
	    '''
	    attention_scores_normalized = np.zeros((len(barcode_info),len(barcode_info)))
	    for index in range (0, X_attention_bundle[0].shape[1]):
	        i = X_attention_bundle[0][0][index]
	        j = X_attention_bundle[0][1][index]
	        attention_scores_normalized [i][j] = X_attention_bundle[1][index][0]
	    ##############
	    adjacency_matrix = np.zeros((len(barcode_info),len(barcode_info)))
	    for index in range (0, X_attention_bundle[0].shape[1]):
	        i = X_attention_bundle[0][0][index]
	        j = X_attention_bundle[0][1][index]
	        adjacency_matrix [i][j] = 1
	    '''
	
	    ##############
	    
	    #hold_attention_score = copy.deepcopy(attention_scores)  
	    #attention_scores = copy.deepcopy(hold_attention_score)  
	    ####################################################################################
	   
	    ###########################
	    
	    ccc_index_dict = dict()
	    threshold_down =  np.percentile(sorted(distribution), 0)
	    threshold_up =  np.percentile(sorted(distribution), 100)
	    connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
	    for j in range (0, datapoint_size):
	        #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
	        for i in range (0, datapoint_size):
	            atn_score_list = attention_scores[i][j]
	            #print(len(atn_score_list))
	            #s = min(0,len(atn_score_list)-1)
	            for k in range (0, len(atn_score_list)):
	                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
	                    connecting_edges[i][j] = 1
	                    ccc_index_dict[i] = ''
	                    ccc_index_dict[j] = ''
	
	
	
	    graph = csr_matrix(connecting_edges)
	    n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
	    print('number of component %d'%n_components)
	
	    count_points_component = np.zeros((n_components))
	    for i in range (0, len(labels)):
	         count_points_component[labels[i]] = count_points_component[labels[i]] + 1
	
	    print(count_points_component)
	
	    id_label = 2 # initially all are zero. =1 those who have self edge but above threshold. >= 2 who belong to some component
	    index_dict = dict()
	    for i in range (0, count_points_component.shape[0]):
	        if count_points_component[i]>1:
	            index_dict[i] = id_label
	            id_label = id_label+1
	
	    print(id_label)
	
	
	    for i in range (0, len(barcode_info)):
	    #    if barcode_info[i][0] in barcode_label:
	        if count_points_component[labels[i]] > 1:
	            barcode_info[i][3] = index_dict[labels[i]] #2
	        elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
	            barcode_info[i][3] = 1
	        else:
	            barcode_info[i][3] = 0
	
	    #######################
	
	
	    
	    data_list=dict()
	    data_list['pathology_label']=[]
	    data_list['component_label']=[]
	    data_list['X']=[]
	    data_list['Y']=[]
	
	    for i in range (0, len(barcode_info)):
	        #if barcode_type[barcode_info[i][0]] == 'zero':
	        #    continue
	        data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
	        data_list['component_label'].append(barcode_info[i][3])
	        data_list['X'].append(barcode_info[i][1])
	        data_list['Y'].append(barcode_info[i][2])
	
	
	    data_list_pd = pd.DataFrame(data_list)
	    #set1 = altairThemes.get_colour_scheme("Set1", len(data_list_pd["component_label"].unique()))
	    set1 = altairThemes.get_colour_scheme("Set1", id_label)
	    set1[0] = '#000000'
	
	    chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
	        alt.X('X', scale=alt.Scale(zero=False)),
	        alt.Y('Y', scale=alt.Scale(zero=False)),
	        shape = alt.Shape('pathology_label:N'), #shape = "pathology_label",
	        color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
	        tooltip=['component_label']
	    )#.configure_legend(labelFontSize=6, symbolLimit=50)
	    # output 2
	    save_path = '/cluster/home/t116508uhn/64630/'
	    #chart.save(save_path+args.data_name+'_altair_plot_bothAbove98_3dim_tanh_3heads_l2attention_th95_'+filename[run_time]+'.html')

	    ##############
	    '''
	    region_list =[2, 3, 9, 11, 4, 5, 7]
	    
	    spot_interest_list = []
	    for i in range (0, len(barcode_info)):
	        if data_list['component_label'][i] in region_list:
	            
	            spot_interest_list.append(i)
	    '''
	    ###############
	    csv_record = []
	    csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'attention_score', 'component', 'from_id', 'to_id'])
	    for j in range (0, len(barcode_info)):
	        for i in range (0, len(barcode_info)):
	            
	            if i==j:
	                if len(lig_rec_dict[i][j])==0:
	                    continue
	            
	            atn_score_list = attention_scores[i][j]
	            for k in range (0, len(atn_score_list)):
	                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: 
	                    if barcode_info[i][3]==0:
	                        print('error')
	                    elif barcode_info[i][3]==1:
	                        csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], min_attention_score + attention_scores[i][j][k], '0-single', i, j])
	                    else:
	                        csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], min_attention_score + attention_scores[i][j][k], barcode_info[i][3], i, j])
	    '''
	    df = pd.DataFrame(csv_record)
	    df.to_csv('/cluster/home/t116508uhn/64630/input_test.csv', index=False, header=False)
	    ############
	    alt.themes.register("publishTheme", altairThemes.publishTheme)
	    # enable the newly registered theme
	    alt.themes.enable("publishTheme")
	    inFile = '/cluster/home/t116508uhn/64630/input_test.csv' #sys.argv[1]
	    df = readCsv(inFile)
	    df = preprocessDf(df)
	    outPathRoot = inFile.split('.')[0]
	    p = plot(df)
	    #outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_'+filename[run_time]+'_th99p7_h512_l2attention_'+str(len(csv_record))+'edges.html' #filteredl2attention__ l2attention_
	    outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_'+filename[run_time]+'_selective_only_Tcellzone_th90_h512_'+str(len(csv_record))+'edges.html' #filteredl2attention__ l2attention_
	    p.save(outPath)	# output 3
	    '''
	    ###########	
	    #run = 1
	    #csv_record_dict = defaultdict(list)
	    print('records found %d'%len(csv_record))
	    for i in range (1, len(csv_record)):
	        key_value = str(csv_record[i][6]) +'-'+ str(csv_record[i][7]) + '-' + csv_record[i][2] + '-' + csv_record[i][3]# + '-'  + str( csv_record[i][5])
	        csv_record_dict[key_value].append([csv_record[i][4], run])
	        

for key_value in csv_record_dict.keys():
    run_dict = defaultdict(list)
    for scores in csv_record_dict[key_value]:
        run_dict[scores[1]].append(scores[0])
    
    for runs in run_dict.keys():
        run_dict[runs] = np.mean(run_dict[runs])
        
        
    csv_record_dict[key_value] = []
    for runs in run_dict.keys():
        csv_record_dict[key_value].append([run_dict[runs],runs])
        
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name+'_merged_5runs_allEdges', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump(csv_record_dict, fp)

###### write the combined result in a csv file #####################
csv_record = []
csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'attention_score_run1','attention_score_run2','attention_score_run3','attention_score_run4','attention_score_run5', 'from_id', 'to_id'])
for key_value in csv_record_dict.keys():
    item = key_value.split('-')
    i = int(item[0])
    j = int(item[1])
    ligand = item[2]
    receptor = item[3]   
    dummy_entry = []
    dummy_entry.append(barcode_info[i][0])
    dummy_entry.append(barcode_info[j][0])
    dummy_entry.append(ligand)
    dummy_entry.append(receptor)
    
    
    for k in range (0, len(csv_record_dict[key_value])):
        dummy_entry.append(csv_record_dict[key_value][k][0])
        
    dummy_entry.append(i)
    dummy_entry.append(j)        
        

    csv_record.append(dummy_entry)
        
        
print('total edges count %d'%len(csv_record))
df = pd.DataFrame(csv_record) # output 4
df.to_csv('/cluster/home/t116508uhn/64630/NEST_output_allEdges_5runs_'+args.data_name+'.csv', index=False, header=False)



fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name+'_merged_5runs', 'rb')
csv_record_dict = pickle.load(fp)

######## intersection of multiple runs of the output graph #################
#total_runs = 2
combined_score_distribution = []
csv_record = []
csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'attention_score', 'component', 'from_id', 'to_id'])
csv_record_intersect_dict = defaultdict(dict) 
for key_value in csv_record_dict.keys():
    if len(csv_record_dict[key_value])>=5: #3: #((total_runs*80)/100):
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        ligand = item[2]
        receptor = item[3]        
        ###
        
        score = 0
        for k in range (0, len(csv_record_dict[key_value])):
            score = score + csv_record_dict[key_value][k][0]
        score = score/len(csv_record_dict[key_value]) # take the average score
        ''''''
        ###        
        label = -1 #csv_record_dict[key_value][total_runs-1][1]
        #score = csv_record_dict[key_value][total_runs-1][0] #score/total_runs
        if ligand+'-'+receptor not in csv_record_intersect_dict or label not in csv_record_intersect_dict[ligand+'-'+receptor]:
            csv_record_intersect_dict[ligand+'-'+receptor][label] = []
        
        csv_record_intersect_dict[ligand+'-'+receptor][label].append(score)
        csv_record.append([barcode_info[i][0], barcode_info[j][0], ligand, receptor, score, label, i, j])
        combined_score_distribution.append(score)
        
print('common LR count %d'%len(csv_record))
############################################### ONLY for Human Lymph Node #################################################################
'''
combined_score_distribution_ccl19_ccr7 = []
for k in range (1, len(csv_record)):
    i = csv_record[k][6]
    j = csv_record[k][7]
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    if ligand =='CCL19' and receptor == 'CCR7':
        combined_score_distribution_ccl19_ccr7.append(csv_record[k][4])
        
some_dict = dict(A=combined_score_distribution, B=combined_score_distribution_ccl19_ccr7)
'''
'''
     
    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in some_dict.items()]))

    df = df.rename(columns={'A': 'all_pairs', 'B': 'CCL19_CCR7'})

    source = df

    chart = alt.Chart(source).transform_fold(
        ['all_pairs',
         'CCL19_CCR7'],
        as_ = ['distribution_type', 'value']
    ).transform_density(
        density = 'value',
        bandwidth=0.3,
        groupby=['distribution_type'],        
        counts = True,
        steps=100
    ).mark_area(opacity=0.5).encode(
        alt.X('value:Q'),
        alt.Y('density:Q', stack='zero' ),
        alt.Color('distribution_type:N')
    )#.properties(width=400, height=100)


chart = alt.Chart(source).transform_fold(
    ['all_pairs', 'CCL19_CCR7'],
    as_=['Distribution Type', 'Attention Score']
).mark_bar(
    opacity=0.5,
    binSpacing=0
).encode(
    alt.X('Attention Score:Q', bin=alt.Bin(maxbins=100)),
    alt.Y('count()', stack=None),
    alt.Color('Distribution Type:N')
)

chart.save(save_path+'region_of_interest_filtered_combined_attention_distribution.html')
'''
############################################################################################

threshold_value =  np.percentile(combined_score_distribution, 80) #_ccl19_ccr7
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
csv_record_final = []
csv_record_final.append(csv_record[0])
for k in range (1, len(csv_record)):
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    #if ligand =='CCL19' and receptor == 'CCR7':
    if csv_record[k][4] >= threshold_value:    
        csv_record_final.append(csv_record[k])




for k in range (1, len(csv_record_final)):
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]
    connecting_edges[i][j]=1
        
graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
print('number of component %d'%n_components)

count_points_component = np.zeros((n_components))
for i in range (0, len(labels)):
     count_points_component[labels[i]] = count_points_component[labels[i]] + 1

print(count_points_component)

id_label = 2 # initially all are zero. =1 those who have self edge but above threshold. >= 2 who belong to some component
index_dict = dict()
for i in range (0, count_points_component.shape[0]):
    if count_points_component[i]>1:
        index_dict[i] = id_label
        id_label = id_label+1

print(id_label)

for i in range (0, len(barcode_info)):
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
        barcode_info[i][3] = 1
    else:
        barcode_info[i][3] = 0

# update the label based on new component numbers
#max opacity

for record in range (1, len(csv_record_final)):
    i = csv_record_final[record][6]
    label = barcode_info[i][3]
    csv_record_final[record][5] = label
    
##### save the file for downstream analysis ########
######################################################################################################################## 
## 
i=0
j=0
csv_record_final.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j])
df = pd.DataFrame(csv_record_final) # output 4
df.to_csv('/cluster/home/t116508uhn/64630/NEST_combined_output_'+args.data_name+'.csv', index=False, header=False)

###########	list those spots who are participating in CCC ##################
filename_str = 'NEST_combined_output_'+args.data_name+'.csv'
inFile = '/cluster/home/t116508uhn/64630/'+filename_str #'/cluster/home/t116508uhn/64630/input_test.csv' #sys.argv[1]
df = pd.read_csv(inFile, sep=",")
csv_record_final = df.values.tolist()
df_column_names = list(df.columns)
csv_record_final = [df_column_names] + csv_record_final

active_spot = defaultdict(list)
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    record = csv_record_final[record_idx]
    i = record[6]
    pathology_label = barcode_type[barcode_info[i][0]]
    component_label = record[5]
    X = barcode_info[i][1]
    Y = -barcode_info[i][2]
    opacity = record[4]
    active_spot[i].append([pathology_label, component_label, X, Y, opacity])
    
    j = record[7]
    pathology_label = barcode_type[barcode_info[j][0]]
    component_label = record[5]
    X = barcode_info[j][1]
    Y = -barcode_info[j][2]
    opacity = record[4]   
    active_spot[j].append([pathology_label, component_label, X, Y, opacity])
    ''''''
    
######### color the spots in the plot with opacity = attention score #################
opacity_list = []
for i in active_spot:
    sum_opacity = []
    for edges in active_spot[i]:
        sum_opacity.append(edges[4])
        
    avg_opacity = np.max(sum_opacity) #np.mean(sum_opacity)
    opacity_list.append(avg_opacity) 
    active_spot[i]=[active_spot[i][0][0], active_spot[i][0][1], active_spot[i][0][2], active_spot[i][0][3], avg_opacity]

min_opacity = np.min(opacity_list)
max_opacity = np.max(opacity_list)
min_opacity = min_opacity - 5
#######################################################################################
data_list=dict()
data_list['pathology_label']=[]
data_list['component_label']=[]
data_list['X']=[]
data_list['Y']=[]   
data_list['opacity']=[] 

for i in range (0, len(barcode_info)):        
    if i in active_spot:
        data_list['pathology_label'].append(active_spot[i][0])
        data_list['component_label'].append(active_spot[i][1])
        data_list['X'].append(active_spot[i][2])
        data_list['Y'].append(active_spot[i][3])
        data_list['opacity'].append((active_spot[i][4]-min_opacity)/(max_opacity-min_opacity))
        
    else:
        data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
        data_list['component_label'].append(0)
        data_list['X'].append(barcode_info[i][1])
        data_list['Y'].append(-barcode_info[i][2])
        data_list['opacity'].append(0.1)


#id_label= len(list(set(data_list['component_label'])))
data_list_pd = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", id_label)
set1[0] = '#000000'
chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    shape = alt.Shape('pathology_label:N'), #shape = "pathology_label",
    color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
    #opacity=alt.Opacity('opacity:N'), #"opacity",
    tooltip=['component_label'] #,'opacity'
)#.configure_legend(labelFontSize=6, symbolLimit=50)

# output 6
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'altair_plot_test.html')
####################################################################################################################
filename_str = 'NEST_combined_output_'+args.data_name+'.csv'
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")
inFile = '/cluster/home/t116508uhn/64630/'+filename_str #'/cluster/home/t116508uhn/64630/input_test.csv' #sys.argv[1]
df = readCsv(inFile)
df = preprocessDf(df)
outPathRoot = inFile.split('.')[0]
p = plot(df)
outPath = '/cluster/home/t116508uhn/64630/test_hist_temp.html'
p.save(outPath)	# output 5
#####################################################################################################################




##################################################
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
set1 = altairThemes.get_colour_scheme("Set1", id_label)
colors = set1
colors[0] = '#000000'
ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(barcode_info)):    
    ids.append(i)
    x_index.append(barcode_info[i][1])
    y_index.append(barcode_info[i][2])    
    colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)


from pyvis.network import Network
import networkx as nx

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if 'tumor'in pathologist_label[i][1]: #'Tumour':
        barcode_type[pathologist_label[i][0]] = 1
    else:
        barcode_type[pathologist_label[i][0]] = 0
    '''
    elif pathologist_label[i][1] == 'stroma_deserted':
        barcode_type[pathologist_label[i][0]] = 0
    elif pathologist_label[i][1] =='acinar_reactive':
        barcode_type[pathologist_label[i][0]] = 2
    else:
        barcode_type[pathologist_label[i][0]] = 'zero' #0
    '''
g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    label_str =  str(i)+'_c:'+str(barcode_info[i][3])+'_'
    #if barcode_type[barcode_info[i][0]] == 'zero':
    #    continue
    if barcode_type[barcode_info[i][0]] == 0: #stroma
        marker_size = 'circle'
        label_str = label_str + 'stroma'
    elif barcode_type[barcode_info[i][0]] == 1: #tumor
        marker_size = 'box'
        label_str = label_str + 'tumor'
    else:
        marker_size = 'ellipse'
        label_str = label_str + 'acinar_reactive'
	
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

count_edges = 0
for k in range (1, len(csv_record_final)):
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]    
    ligand = csv_record_final[k][2]
    receptor = csv_record_final[k][3]
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record_final[k][4]
    g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score), color=colors_point[i] ) 
    count_edges = count_edges + 1
     
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/test_interactive.dot")

