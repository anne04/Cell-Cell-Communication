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

alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")

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
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/LUAD/LUAD_GSM5702473_TD1/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='LUAD_GSM5702473_TD1', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_3attr', help='model name')
#parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()
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

	
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(temp.X)))  
adata_X = np.transpose(temp)  
#adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X
'''
#############################################################    
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V1_Human_Lymph_Node_spatial', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()
'''
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()
filter_min_cell = 5
threshold_expression = 98
'''
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/exp1/exp1_C1/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_140694', help='The name of dataset')
#parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
#parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()
filter_min_cell = 1
threshold_expression = 98


'''
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/exp2_B1/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_130355_B1', help='The name of dataset')
args = parser.parse_args()
filter_min_cell = 5
threshold_expression = 98.1
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/exp2_A1/outs/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='PDAC_130355_A1', help='The name of dataset')
args = parser.parse_args()
filter_min_cell = 5
threshold_expression = 98.7

####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
print(data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)

#sc.pp.log1p(adata_h5)

sc.pp.filter_genes(adata_h5, min_cells=filter_min_cell)
print(adata_h5)

#sc.pp.highly_variable_genes(adata_h5) #3952
#adata_h5 = adata_h5[:, adata_h5.var['highly_variable']]
#print(adata_h5)

gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
cell_barcode = np.array(adata_h5.obs.index)
#barcode_info.append("")
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)
#adata_X=sparse.csr_matrix.toarray(adata_h5.X)
#adata_X = sc.pp.scale(adata_X)
cell_vs_gene = copy.deepcopy(adata_X)
print('min value %g'%np.min(cell_vs_gene))


#######################################################

## find PLXNB2 gene id
for j in range (0, len(gene_ids)):
    if gene_ids[j]=='SEMA4C': #'PLXNB2':
        print(j)
        break

i=0
barcode_info=[]
for cell_code in cell_barcode:
    #scaled_count = (max_count-cell_vs_gene[i,j])/(max_count-min_count)
    barcode_info.append([cell_code, coordinates[i,0], coordinates[i,1], cell_vs_gene[i,j]])
    i=i+1
    
data_list=dict()
data_list['X']=[]
data_list['Y']=[]   
data_list['gene_expression']=[] 

for i in range (0, len(barcode_info)):
    data_list['X'].append(barcode_info[i][1])
    data_list['Y'].append(-barcode_info[i][2])
    data_list['gene_expression'].append(barcode_info[i][3])


source= pd.DataFrame(data_list)

chart = alt.Chart(source).mark_point(filled=True).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    color=alt.Color('gene_expression:Q', scale=alt.Scale(scheme='magma'))
)
save_path = '/cluster/home/t116508uhn/64630/'
#chart.save(save_path+'altair_plot_sema4c.html')


########################################################
 
i=0
barcode_serial = dict()
for cell_code in cell_barcode:
    barcode_serial[cell_code]=i
    i=i+1
    
i=0
barcode_info=[]
for cell_code in cell_barcode:
    barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1],0])
    i=i+1
	
i=0
node_id_sorted_xy=[]
for cell_code in cell_barcode:
    node_id_sorted_xy.append([i, coordinates[i,0],coordinates[i,1]])
    i=i+1
	
node_id_sorted_xy = sorted(node_id_sorted_xy, key = lambda x: (x[1], x[2]))

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name+'_'+'node_id_sorted_xy', 'wb') as fp:  #b, a:[0:5]   ##
#	pickle.dump(node_id_sorted_xy, fp)


#################### 
'''
gene_vs_cell = np.transpose(cell_vs_gene)  
np.save("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_quantile_transformed_"+args.data_name, gene_vs_cell)
df = pd.DataFrame(gene_ids)
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/gene_ids_'+args.data_name+'.csv', index=False, header=False)
df = pd.DataFrame(cell_barcode)
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cell_barcode_'+args.data_name+'.csv', index=False, header=False)
'''   

#cell_vs_gene_scaled = sc.pp.scale(adata_X) # rows = cells, columns = genes
####################
'''
for i in range (0, cell_vs_gene.shape[0]):
    max_value = np.max(cell_vs_gene[i][:])
    min_value = np.min(cell_vs_gene[i][:])
    for j in range (0, cell_vs_gene.shape[1]):
        cell_vs_gene[i][j] = (cell_vs_gene[i][j]-min_value)/(max_value-min_value)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'cell_vs_gene_quantile_transformed_scaled', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump(cell_vs_gene, fp)
'''
#
####################
'''
adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
#adata_X = sc.pp.scale(adata_X)
#adata_X = sc.pp.pca(adata_X, n_comps=args.Dim_PCA)
cell_vs_gene = sparse.csr_matrix.toarray(adata_X) #adata_X
'''
####################
####################
'''
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    cell_percentile.append([np.percentile(sorted(cell_vs_gene_scaled[i]), 10), np.percentile(sorted(cell_vs_gene_scaled[i]), 20),np.percentile(sorted(cell_vs_gene_scaled[i]), 70), np.percentile(sorted(cell_vs_gene_scaled[i]), 97)])
'''


gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
	
'''gene_marker_ids = dict()
gene_marker_file = '/cluster/home/t116508uhn/64630/Geneset_22Sep21_Subtypesonly_edited.csv'
df = pd.read_csv(gene_marker_file)
for i in range (0, df["Name"].shape[0]):
    if df["Name"][i] in gene_info:
        gene_marker_ids[df["Name"][i]] = ''
'''

ligand_dict_dataset = defaultdict(list)
cell_cell_contact = dict()
'''
OMNIPATH_file = '/cluster/home/t116508uhn/64630/omnipath_records_2023Feb.csv'   
df = pd.read_csv(OMNIPATH_file)
for i in range (0, df['genesymbol_intercell_source'].shape[0]):
    
    ligand = df['genesymbol_intercell_source'][i]
    if 'ligand' not in  df['category_intercell_source'][i]:
        continue
    if ligand not in gene_info:
        continue
        
    receptor = df['genesymbol_intercell_target'][i]
    if 'receptor' not in df['category_intercell_target'][i]:
        continue
    if receptor not in gene_info:
        continue
    ligand_dict_dataset[ligand].append(receptor)
    if df['category_intercell_source'][i] == 'cell_surface_ligand':
        cell_cell_contact[ligand] = ''
'''    
   
cell_chat_file = '/cluster/home/t116508uhn/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    #if ligand not in gene_marker_ids:
    if ligand not in gene_info:
        continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
        #if receptor in gene_marker_ids:
            ligand_dict_dataset[ligand].append(receptor)
            #######
            if df["annotation"][i] == 'Cell-Cell Contact':
                cell_cell_contact[receptor] = ''
            #######                
            
print(len(ligand_dict_dataset.keys()))

nichetalk_file = '/cluster/home/t116508uhn/NicheNet-LR-pairs.csv'   
df = pd.read_csv(nichetalk_file)
for i in range (0, df["from"].shape[0]):
    ligand = df["from"][i]
    #if ligand not in gene_marker_ids:
    if ligand not in gene_info:
        continue
    receptor = df["to"][i]
    #if receptor not in gene_marker_ids:
    if receptor not in gene_info:
        continue
    ligand_dict_dataset[ligand].append(receptor)
    
##############################################################
print('number of ligands %d '%len(ligand_dict_dataset.keys()))
count_pair = 0
for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    gene_info[gene] = 'included'
    for receptor_gene in ligand_dict_dataset[gene]:
        gene_info[receptor_gene] = 'included'
        count_pair = count_pair + 1
        
print('number of pairs %d '%count_pair)       

count = 0
included_gene=[]
for gene in gene_info.keys(): 
    if gene_info[gene] == 'included':
        count = count + 1
        included_gene.append(gene)
        
print('number of affected genes %d '%count)
affected_gene_count = count
######################################

lr_gene_index = []
for gene in gene_info.keys(): 
    if gene_info[gene] == 'included':
        lr_gene_index.append(gene_index[gene])

lr_gene_index = sorted(lr_gene_index)
cell_vs_lrgene = cell_vs_gene[:, lr_gene_index]
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'cell_vs_lrgene_quantile_transformed_'+args.data_name, 'wb') as fp:  #b, a:[0:5]   
	pickle.dump(cell_vs_lrgene, fp)
	
''''''
######################################

ligand_list = list(ligand_dict_dataset.keys())  
print('len ligand_list %d'%len(ligand_list))
total_relation = 0
l_r_pair = dict()
count = 0
lr_id = 0
for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    l_r_pair[gene] = dict()
    for receptor_gene in ligand_dict_dataset[gene]:
        l_r_pair[gene][receptor_gene] = lr_id 
        lr_id  = lr_id  + 1
        
print('total type of l-r pairs found: %d'%lr_id )

#id_list = []
#for receptor_gene in l_r_pair['CCL19']:
#    id_list.append(l_r_pair['CCL19'][receptor_gene])

from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
for j in range(0, distance_matrix.shape[1]):
    max_value=np.max(distance_matrix[:,j])
    min_value=np.min(distance_matrix[:,j])
    for i in range(distance_matrix.shape[0]):
        dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)
        	
    #list_indx = list(np.argsort(dist_X[:,j]))
    #k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
    for i in range(0, distance_matrix.shape[0]):
        if distance_matrix[i,j] > spot_diameter*4: #i not in k_higher:
            dist_X[i,j] = 0 #-1
            
cell_rec_count = np.zeros((cell_vs_gene.shape[0]))

########
######################################
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    y = sorted(cell_vs_gene[i])
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='increasing')
    kn_value = y[kn.knee-1]
    #print(y[kn.knee-1])
    cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 90), np.percentile(y, threshold_expression), kn_value])


'''
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    #print(np.histogram(cell_vs_gene[i]))
    y = np.histogram(cell_vs_gene[i])[0] # density: 
    x = range(0, len(y))
    #print(np.histogram(cell_vs_gene[i])[0])

    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    kn_value = np.histogram(cell_vs_gene[i])[1][kn.knee]
    #print('%d'%(kn.knee ))
    cell_percentile.append([np.percentile(cell_vs_gene[i], 10), np.percentile(cell_vs_gene[i], 20),np.percentile(cell_vs_gene[i], 95), np.percentile(cell_vs_gene[i], 98), kn_value])
'''
#gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
'''
gene_percentile = dict()
for i in range (0, cell_vs_gene.shape[1]):
    y = np.histogram(cell_vs_gene[:,i])[0]
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    kn_value = np.histogram(cell_vs_gene[:,i])[1][kn.knee-1]
    gene_percentile[gene_ids[i]] = [np.percentile(cell_vs_gene[:,i], 10), np.percentile(cell_vs_gene[:,i], 50),np.percentile(cell_vs_gene[:,i], 80), np.percentile(cell_vs_gene[:,i], 97), kn_value]
'''
##############################################################################
count_total_edges = 0
activated_cell_index = dict()

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
    
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
start_index = 0 #args.slice
end_index = len(ligand_list) #min(len(ligand_list), start_index+100)
included_LR = defaultdict(dict)
for g in range(start_index, end_index): 
    gene = ligand_list[g]
    for i in range (0, cell_vs_gene.shape[0]): # ligand
        count_rec = 0    
        if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i][3]:
            continue
        
        for j in range (0, cell_vs_gene.shape[0]): # receptor
            if distance_matrix[i,j] > spot_diameter*4:
                continue

            #if gene in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
            #    continue

            for gene_rec in ligand_dict_dataset[gene]:
                if cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j][3]: # or cell_vs_gene[i][gene_index[gene]] >= cell_percentile[i][4] :#gene_list_percentile[gene_rec][1]: #global_percentile: #
                    if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                        continue

                    '''if gene_rec in cell_cell_contact and distance_matrix[i,j] < spot_diameter:
                        print(gene)'''

                    communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                    '''if gene=='L1CAM':
                        count = count+1
                    elif gene=='LAMC2':
                        count2 = count2+1'''
                    '''
                    if l_r_pair[gene][gene_rec] == -1: 
                        l_r_pair[gene][gene_rec] = pair_id
                        pair_id = pair_id + 1 
                    '''
                    relation_id = l_r_pair[gene][gene_rec]
                    #print("%s - %s "%(gene, gene_rec))
                    if communication_score<=0:
                        print('zero valued ccc score found')
                        continue	
                    cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
                    included_LR[gene][gene_rec] = ''
                    count_rec = count_rec + 1
                    count_total_edges = count_total_edges + 1
                    activated_cell_index[i] = ''
                    activated_cell_index[j] = ''

                            
        cell_rec_count[i] =  count_rec   
        #print("%d - %d "%(i, count_rec))
        #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))
    
    print(g)
    
print('total number of edges in the input graph %d '%count_total_edges)

 

'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'omnipath_communication_scores_allPair_bothAboveDensity', 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor], fp) #a - [0:5]
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'omnipath_communication_scores_threshold_distance_bothAboveDensity', 'wb') as fp: #b, b_1, a
    pickle.dump(cells_ligand_vs_receptor, fp) #a - [0:5]

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'omnipath_communication_scores_allPair_bothAboveCellKnee', 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor], fp) #a - [0:5]
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'omnipath_communication_scores_threshold_distance_eitherAboveCellKnee', 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor], fp) #a - [0:5]
'''
############################################################
	
'''    
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')	
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

slice = -30
while slice < 544:
    slice = slice + 30
    print('read %d'%slice)
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_selective_lr_STnCCC_c_'+str(slice), 'rb') as fp: #b, b_1, a
        cells_ligand_vs_receptor_temp, l_r_pair, ligand_list, activated_cell_index = pickle.load(fp) 
        
    for i in range (0, len(cells_ligand_vs_receptor)):
        for j in range (0, len(cells_ligand_vs_receptor)):
            if len(cells_ligand_vs_receptor_temp[i][j])>0:
                cells_ligand_vs_receptor[i][j].extend(cells_ligand_vs_receptor_temp[i][j]) 
'''
###############################

################################################################################
ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
#local_list = np.zeros((102))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    # above 5th percentile only
                    #if cell_vs_gene[i][gene_index[gene]] >= cell_percentile[i][2] and cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j][2]:
                    count_edge = count_edge + 1
                    count_local = count_local + 1
#print(count_edge)                      
                    mean_ccc = cells_ligand_vs_receptor[i][j][k][2]
                    row_col.append([i,j])
                    #if gene=='SERPINA1': # or gene=='MIF':
                    #    ccc_index_dict[i] = ''
                    #ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc,cells_ligand_vs_receptor[i][j][k][3]])
                    #edge_weight.append([dist_X[i,j], mean_ccc])
                    lig_rec.append([gene, gene_rec])                      
                
                if max_local < count_local:
                    max_local = count_local
            '''
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
            '''
            #local_list[count_local] = local_list[count_local] + 1

print('len row col %d'%len(row_col))
print('count local %d'%max_local) 



##########
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell99th', 'wb') as fp:  #b, a:[0:5]   
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_3d', 'wb') as fp:  #b, a:[0:5]  _filtered 
    pickle.dump([row_col, edge_weight, lig_rec], fp)
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_cell_vs_gene_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   _filtered
	pickle.dump(cell_vs_gene, fp)

##########
for lr_pair in lig_rec:
    gene=lr_pair[0]
    rec_gene = lr_pair[1]
    l_r_pair[gene][rec_gene] = '*'
    
existing_lr_pair=[]
for gene in l_r_pair:
    for rec_gene in l_r_pair[gene]:
        if l_r_pair[gene][rec_gene] == '*':
            existing_lr_pair.append(gene+'-'+rec_gene)
   
##########

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_kneepoint_woBlankedge', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_kneepoint', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th', 'wb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th', 'wb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_density_kneepoint', 'wb') as fp:  #b, a:[0:5]   
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_omniPath_separate_'+'threshold_distance_density_kneepoint', 'wb') as fp:  #b, a:[0:5]   
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'eitherOne_density_kneepoint', 'rb') as fp:  #b, a:[0:5]   
    pickle.dump([row_col, edge_weight, lig_rec], fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'cell_vs_gene_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump(cell_vs_gene, fp)
	
edge_list = []
lig_rec_list = []
row_col_list = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    ligand_gene = lig_rec[index][0]
    receptor_gene = lig_rec[index][1]
    k = l_r_pair[ligand_gene][receptor_gene]
    
    if edge_weight[index][1] > 0:
        edge_list.append([edge_weight[index][0], edge_weight[index][1], k])
        lig_rec_list.append([ligand_gene, receptor_gene])
        row_col_list.append([i,j])
    
edge_weight = edge_list
row_col = row_col_list
lig_rec = lig_rec_list

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_3d', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump([row_col, edge_weight, lig_rec], fp)
#####################################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'cell_vs_gene_quantile_transformed', 'rb') as fp:
    cell_vs_gene = pickle.load(fp)


data_list=defaultdict(list)
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[1]):
        data_list[cell_barcode[i]].append(cell_vs_gene[i][j])
        
        
data_list_pd = pd.DataFrame(data_list)    
gene_name = []
for j in range (0, cell_vs_gene.shape[1]):
    gene_name.append(gene_ids[j])
    
data_list_pd[' ']=gene_name   
data_list_pd = data_list_pd.set_index(' ')    
data_list_pd.to_csv('/cluster/home/t116508uhn/PDAC_64630_gene_vs_cell.csv')
	
	
	
	
################################################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th', 'rb') as fp:  #b, a:[0:5]   
    row_col, edge_weight, lig_rec = pickle.load(fp)

max_value = -1000
min_value = 10000
for index in range (0, len(edge_weight)):
    if edge_weight[index][1] > max_value:
        max_value = edge_weight[index][1]
    if edge_weight[index][1] < min_value:
        min_value = edge_weight[index][1]
        
for index in range (0, len(edge_weight)):
    edge_weight[index][1] = 0.1 + ((edge_weight[index][1] - min_value)/(max_value-min_value))*(1-0.1)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_scaled', 'wb') as fp:  #b, a:[0:5]   
    pickle.dump([row_col, edge_weight, lig_rec], fp)

########################################################### Visualization starts ##################
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

#############################################################################
ccc_too_many_cells_LUAD = pd.read_csv('/cluster/projects/schwartzgroup/fatema/CCST/exp2_D1_ccc_toomanycells_cluster.csv')
ccc_too_many_cells_LUAD_dict = dict()
for i in range(0, len(ccc_too_many_cells_LUAD)):
    ccc_too_many_cells_LUAD_dict[ccc_too_many_cells_LUAD['cell'][i]] = int(ccc_too_many_cells_LUAD['cluster'][i])

for i in range(0, len(barcode_info)):
    barcode_info[i][3] = ccc_too_many_cells_LUAD_dict[barcode_info[i][0]]
	

barcode_type=dict()
for i in range (0, len(barcode_info)):
    barcode_type[barcode_info[i][0]] = 0 
        
#####

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


#######################################################################
cell_expression = defaultdict(list)
for index in range (0, len(row_col)):
    ligand = lig_rec[index][0]
    rec = lig_rec[index][1]
    if ligand=='PLXNB2' and rec=='MET':
        i = row_col[index][0]
        j = row_col[index][1]        
        ccc = edge_weight[index][1]
        cell_expression[i].append(ccc)
        cell_expression[j].append(ccc)
        
i=0
barcode_info=[]
expression_distribution = []
for cell_code in cell_barcode:
    if len(cell_expression[i])==0:
        cell_expression[i].append(0)
        
    barcode_info.append([cell_code, coordinates[i,0], coordinates[i,1], np.max(cell_expression[i])])
    expression_distribution.append(np.max(cell_expression[i]))
    i=i+1

min_count = np.min(expression_distribution)
max_count = np.max(expression_distribution)
for i in range (0, len(barcode_info)):
    barcode_info[i][3] = (max_count-barcode_info[i][3])/(max_count-min_count)

data_list=dict()
data_list['X']=[]
data_list['Y']=[]   
data_list['component_label']=[] 
data_list['opacity']=[] 

for i in range (0, len(barcode_info)):
    data_list['X'].append(barcode_info[i][1])
    data_list['Y'].append(-barcode_info[i][2])
    data_list['component_label'].append(1)
    data_list['opacity'].append(barcode_info[i][3])


data_list_pd = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", 1)
chart = alt.Chart(data_list_pd).mark_point(filled=True).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
    opacity=alt.Opacity('opacity:N'), #"opacity",
    tooltip=['opacity'] 
)#.configure_legend(labelFontSize=6, symbolLimit=50)

# output 6
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'altair_plot_plxnb2.html')
    
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
filename = ["r1_", "r2_", "r3_", "r4_", "r5_", "r6_", "r7_", "r8_", "r9_", "r10_"]
total_runs = 5
start_index = 5
csv_record_dict = defaultdict(list)
for run_time in range (start_index, start_index+total_runs):
    gc.collect()
    #run_time = 2
    run = run_time
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
	    
	    print('min attention score %g'%min_attention_score)
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
	    threshold_down =  np.percentile(sorted(distribution), 90)
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
	
	
	    '''
	    barcode_type=dict()
	    for i in range (1, len(pathologist_label)):
	        if pathologist_label[i][1] == 'tumor': #'Tumour':
	            barcode_type[pathologist_label[i][0]] = '2_tumor'
	        elif pathologist_label[i][1] =='stroma_deserted':
	            barcode_type[pathologist_label[i][0]] = '0_stroma_deserted'
	        elif pathologist_label[i][1] =='acinar_reactive':
	            barcode_type[pathologist_label[i][0]] = '1_acinar_reactive'
	        else:
	            barcode_type[pathologist_label[i][0]] = 'zero' #0
	    '''
	    
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
	    #data_list_pd.to_csv('/cluster/home/t116508uhn/64630/omnipath_ccc_th95_tissue_plot_withFeature_woBlankEdges.csv', index=False)
	    #df_test = pd.read_csv('/cluster/home/t116508uhn/64630/omnipath_ccc_th95_tissue_plot_withFeature_woBlankEdges.csv')
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
	    #chart.save(save_path+args.data_name+'_filtered_CCL19_CCR7_input_graph.html') #
	    #chart.save(save_path+args.data_name+'_CCL19_CCR7_th95_graph.html') #selective_
	    #chart.save(save_path+args.data_name+'_IL21_IL21R_attention_only_th95_l2attention_'+filename[run_time]+'.html') #
	    #chart.save(save_path+args.data_name+'_CCL19_CCR7_attention_only_th95_l1attention_'+filename[run_time]+'.html') #
	    #chart.save(save_path+args.data_name+'_altair_plot_bothAbove98_th99p9_3dim_tanh_h512_l2attention_'+filename[run_time]+'.html') #filtered_l2attention_
	    #chart.save(save_path+'altair_plot_98th_bothAbove98_3dim_tanh_h2048_'+filename[run_time]+'.html')
	    #chart.save(save_path+'altair_plot_bothAbove98_3dim_'+filename[run_time]+'.html')
	    #chart.save(save_path+'altair_plot_97th_bothAbove98_3d_input.html')
	    #chart.save(save_path+'altair_plot_97th_bothAbove98_'+filename[run_time]+'.html')
	    #chart.save(save_path+'region_of_interest_r1.html')
	    #chart.save(save_path+'altair_plot_95_withlrFeature_bothAbove98_'+filename[run_time]+'.html')
	    #chart.save(save_path+'altair_plot_'+'80'+'th_'+filename[run_time]+'.html')
	    #chart.save(save_path+'altair_plot_'+'80'+'th_'+filename[run_time]+'.html')
	    
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
	    '''
	    for j in range (0, len(barcode_info)):
	        for i in range (0, len(barcode_info)):
	            for k in range (0, len(lig_rec_dict[i][j])):
	                csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], -1, barcode_info[i][3], i, j])
	
	    '''
	    
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
        
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name+'_merged_5runs', 'wb') as fp:  #b, a:[0:5]   
	pickle.dump(csv_record_dict, fp)
	
fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name+'_merged_5runs', 'rb')
csv_record_dict = pickle.load(fp)

# intersection 
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


threshold_value =  np.percentile(combined_score_distribution,80) #_ccl19_ccr7
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
for k in range (1, len(csv_record)):
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    #if ligand =='CCL19' and receptor == 'CCR7':
    if csv_record[k][4] >= threshold_value:        
        i = csv_record[k][6]
        j = csv_record[k][7]
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
#    if barcode_info[i][0] in barcode_label:
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
        barcode_info[i][3] = 1
    else:
        barcode_info[i][3] = 0

# update the label based on new component numbers
#max opacity

for record in range (1, len(csv_record)):
    i = csv_record[record][6]
    label = barcode_info[i][3]
    csv_record[record][5] = label
    

###########	

exist_spot = defaultdict(list)
for record_idx in range (1, len(csv_record)):
    record = csv_record[record_idx]
    i = record[6]
    pathology_label = barcode_type[barcode_info[i][0]]
    component_label = record[5]
    X = barcode_info[i][1]
    Y = -barcode_info[i][2]
    opacity = record[4]
    exist_spot[i].append([pathology_label, component_label, X, Y, opacity])
    
    j = record[7]
    pathology_label = barcode_type[barcode_info[j][0]]
    component_label = record[5]
    X = barcode_info[j][1]
    Y = -barcode_info[j][2]
    opacity = record[4]   
    exist_spot[j].append([pathology_label, component_label, X, Y, opacity])
    ''''''
    
opacity_list = []
for i in exist_spot:
    sum_opacity = []
    for edges in exist_spot[i]:
        sum_opacity.append(edges[4])
        
    avg_opacity = np.max(sum_opacity) #np.mean(sum_opacity)
    opacity_list.append(avg_opacity)
    
    exist_spot[i]=[exist_spot[i][0][0], exist_spot[i][0][1], exist_spot[i][0][2], exist_spot[i][0][3], avg_opacity]

min_opacity = np.min(opacity_list)
max_opacity = np.max(opacity_list)
min_opacity = min_opacity - 5

data_list=dict()
data_list['pathology_label']=[]
data_list['component_label']=[]
data_list['X']=[]
data_list['Y']=[]   
data_list['opacity']=[] 

for i in range (0, len(barcode_info)):
    #if barcode_type[barcode_info[i][0]] == 'zero':
    #    continue
        
    if i in exist_spot:
        data_list['pathology_label'].append(exist_spot[i][0])
        data_list['component_label'].append(exist_spot[i][1])
        data_list['X'].append(exist_spot[i][2])
        data_list['Y'].append(exist_spot[i][3])
        data_list['opacity'].append((exist_spot[i][4]-min_opacity)/(max_opacity-min_opacity))
        
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
chart.save(save_path+'altair_plot_'+args.data_name+'_opacity_bothAbove98_th97_90_3dim_tanh_h512_l1l2attention_combined_5runs_'+str(len(csv_record))+'edges.html')
#chart.save(save_path+'altair_plot_'+args.data_name+'_opacity_bothAbove98_th98_3dim_tanh_h512_l1l2attention_'+filename[run_time]+str(len(csv_record))+'edges.html')  #l2attention_

#chart.save(save_path+'altair_plot_'+args.data_name+'_opacity_bothAbove98_th99p7_3dim_tanh_h512_l2attention_combined_'+str(total_runs)+'runs_'+str(len(csv_record))+'edges_100percent.html')  #l2attention_
#chart.save(save_path+'altair_plot_'+args.data_name+'_opacity_bothAbove98_th90_3dim_tanh_h512_filtered_l2attention_combined_'+str(total_runs)+'runs_'+str(len(csv_record))+'edges_100percent.html')  #l2attention_
#chart.save(save_path+'altair_plot_140694_bothAbove98_th99p5_3dim_combined_'+str(total_runs)+'runs_'+str(len(csv_record))+'edges_5.html')  
#chart.save(save_path+'altair_plot_140694_bothAbove98_th98_3dim_combined_'+str(total_runs)+'runs_'+str(len(csv_record))+'edges.html')  
#chart.save(save_path+'altair_plot_140694_bothAbove98_th99p5_3dim_combined_'+str(total_runs)+'runs'.html')  
########################################################################################################################
threshold_value =  np.percentile(combined_score_distribution,0)
csv_record_temp = []
csv_record_temp.append(csv_record[0])
for k in range (1, len(csv_record)):
    if csv_record[k][4] >= threshold_value:    
        csv_record_temp.append(csv_record[k])
   

#for k in range (1, len(csv_record_temp)):
#    csv_record_temp[k][5] = ccc_too_many_cells_LUAD_dict[csv_record_temp[k][0]]
        
i=0
j=0
csv_record_temp.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j])
df = pd.DataFrame(csv_record_temp) # output 4
#df.to_csv('/cluster/home/t116508uhn/64630/input_test_'+args.data_name+'_h512_filtered_l2attention_edges'+str(len(csv_record))+'_combined_th90_100percent_totalruns_'+str(total_runs)+'.csv', index=False, header=False) #
#df.to_csv('/cluster/home/t116508uhn/64630/input_test_'+args.data_name+'_h512_l2attention_edges'+str(len(csv_record))+'_combined_th98p5_100percent_totalruns_'+str(total_runs)+'.csv', index=False, header=False) #
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
outPath = '/cluster/home/t116508uhn/64630/test_hist_temp.html'
#outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_th97_90_h512_l1l2attention_combined_5runs_edges'+str(len(csv_record))+'.html' #l2attention_
#outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_th90_70_67_h512_l1l2attention_'+filename[run_time]+'_edges'+str(len(csv_record))+'.html' #l2attention_
#outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_h512_l2attention_combined_th95_100percent_totalruns_'+str(total_runs)+'_edges'+str(len(csv_record))+'.html' #l2attention_
#outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_h512_filtered_l2attention_combined_th90_100percent_totalruns_'+str(total_runs)+'.html' #l2attention_
p.save(outPath)	# output 5
##########################

#set1[0] = '#000000'
data_list=dict()
data_list['ligand-receptor']=[]
data_list['component_label']=[]
data_list['score']=[]
for lr in csv_record_intersect_dict:
    for component in csv_record_intersect_dict[lr]:
        score = len(csv_record_intersect_dict[lr][component]) #np.sum(csv_record_intersect_dict[lr][component]) #len(csv_record_intersect_dict[lr][component])*np.sum(csv_record_intersect_dict[lr][component]) #/len(csv_record_intersect_dict[lr][component])
        data_list['ligand-receptor'].append(lr)
        data_list['component_label'].append(component)
        data_list['score'].append(score)
    
source = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", len(set(data_list['component_label'])))
chart = alt.Chart(source).mark_bar().encode(
    x=alt.X("ligand-receptor:N", sort='-y', axis=alt.Axis(labelAngle=45)), 
    y='score',
    color=alt.Color("component_label:N", scale = alt.Scale(range=set1)),
    order=alt.Order("component_label", sort="ascending"),
    tooltip=["component_label"]
)        
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'test.html')
       
        
###

df = pd.DataFrame(csv_record)
df.to_csv('/cluster/home/t116508uhn/64630/test_intersection.csv', index=False, header=False)
#df.to_csv('/cluster/home/t116508uhn/64630/ccc_th95_records_bothAbove98th_withlrFeature_intersection.csv', index=False, header=False)

############


################

df = pd.DataFrame(csv_record)
#df.to_csv('/cluster/home/t116508uhn/64630/input_edge_ccc_th95_records_woBlankEdges.csv', index=False, header=False)
df.to_csv('/cluster/home/t116508uhn/64630/ccc_th97_records_woBlankEdges_bothAbove98th.csv', index=False, header=False)

#df.to_csv('/cluster/home/t116508uhn/64630/ccc_th95_omnipath_records_withFeature_woBlankEdges.csv', index=False, header=False)
############################


#############
'''
datapoint_label = []
for i in range (0, datapoint_size):
    if i in ccc_index_dict:
        barcode_info[i][3] = 2
    else:
        barcode_info[i][3] = 0
'''
########
number = 20
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

number = 20
cmap = plt.get_cmap('tab20b')
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

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2


colors_point = []
for i in range (0, len(barcode_info)):      
    colors_point.append(colors[barcode_info[i][3]]) 
  
#cell_count_cluster=np.zeros((labels.shape[0]))
#filltype='none'

#id_label = [0,2] #
#for j in id_label:
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
set1 = altairThemes.get_colour_scheme("Set1", id_label)
colors = set1
colors[0] = '#000000'
for j in range (0, id_label):
    label_i = j
    x_index=[]
    y_index=[]
    marker_size = []
    fillstyles_type = []
    for i in range (0, len(barcode_info)):
        if barcode_type[barcode_info[i][0]]== 'zero':
            continue
        if barcode_info[i][3] == j:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            #cell_count_cluster[j] = cell_count_cluster[j]+1
            spot_color = colors[j]
            if barcode_type[barcode_info[i][0]] == 'stroma_deserted':
                marker_size.append("o") 
                fillstyles_type.append('none') 
                #filltype='none'
            elif barcode_type[barcode_info[i][0]] == 'tumor':
                marker_size.append("^")  
                fillstyles_type.append('full') 
                #filltype = 'full'
            else:
                marker_size.append("*") 
                fillstyles_type.append('none') 
                #filltype = 'none'           
            ###############
    marker_type = []        
    for i in range (0, len(x_index)):  
        marker_type.append(matplotlib.markers.MarkerStyle(marker=marker_size[i]))   
     
    for i in range (0, len(x_index)):  
        plt.scatter(x=x_index[i], y=-y_index[i], label = j, color=colors[j], marker=matplotlib.markers.MarkerStyle(marker=marker_size[i], fillstyle=fillstyles_type[i]), s=15)   
    #filltype = 'full'
    '''
    if len(x_index)>0:
        plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=spot_color, s=15) #marker=marker_size, 
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    '''
#plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 
plt.hist(distribution, color = 'blue',bins = int(len(distribution)/5))
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
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
	
    #g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    
    #g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		#  label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !"
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

threshold_value =  np.percentile(combined_score_distribution,80)
count_edges = 0
for k in range (1, len(csv_record)):
    if csv_record[k][4] < threshold_value:
        continue
    i = csv_record[k][6]
    j = csv_record[k][7]    
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record[k][4]
    g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score), color=colors_point[i] ) 
    count_edges = count_edges + 1
     
'''
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):
            #if attention_scores[i][j][k] >= threshold_down:
            #    #print('hello')
            #key_value_2 = lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            key_value = str(i) +'-'+ str(j) + '-' + lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            if len(csv_record_dict[key_value])>=total_runs:  #key_value_2 in lr_target: #and : 
                edge_score =  min_attention_score + attention_scores[i][j][k]
                title_str =  "L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]#+", "+str(edge_score) #"L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]+", "+str(attention_scores[i][j][k])
                g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) #,width=, arrowsize=int(20),  arrowstyle='fancy'
				# label = title =
'''
#nt.show('mygraph.html')
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/test_interactive.dot")
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_bothAbove98_th95_95_20_3dim_tanh_h512_l1l2attention_split_combined_"+str(total_runs)+"runs_"+str(len(csv_record))+"edges_4of5.dot")
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_bothAbove98_th90_90_80_3dim_tanh_h512_l1l2attention_combined_"+str(total_runs)+"runs_"+str(len(csv_record))+"edges.dot")
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_bothAbove98_th89p5_3dim_tanh_h512_l2attention_combined_"+str(total_runs)+"runs_"+str(len(csv_record))+"edges_100percent.dot")

#write_dot(g, "/cluster/home/t116508uhn/64630/interactive_140694_bothAbove98_3d_th98_leakyRelu.dot")
#


##################

import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
set1 = altairThemes.get_colour_scheme("Set1", id_label)
colors = set1
#colors[0] = '#000000'
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

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    marker_size = 'circle'
    label_str =  str(i)+'_c:'+str(barcode_info[i][3])+'_'
    if barcode_type[barcode_info[i][0]] == 0: 
        label_str = label_str + 'mixed'
    elif barcode_type[barcode_info[i][0]] == 1:
        label_str = label_str + 'Tcell'
    elif barcode_type[barcode_info[i][0]] == 2:
        label_str = label_str + 'B'
    else:
        label_str = label_str + 'GC'

    #g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    
    #g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		#  label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !"
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):    
            #    #print('hello')
            #key_value_2 = lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            key_value = str(i) +'-'+ str(j) + '-' + lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            if len(csv_record_dict[key_value])>=1: #total_runs:  #key_value_2 in lr_target: #and : 
                edge_score =  0 #min_attention_score + attention_scores[i][j][k]
                title_str =  "L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]#+", "+str(edge_score) #"L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]+", "+str(attention_scores[i][j][k])
                g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) #,width=, arrowsize=int(20),  arrowstyle='fancy'
				# label = title =
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_th99p9_"+str(len(csv_record))+"edges.dot")
########################### human lymph node ###########################

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
    #x_index.append(barcode_info[i][1])
    #y_index.append(barcode_info[i][2])  

    x_index.append(barcode_info[i][1]*3)
    y_index.append(barcode_info[i][2]*3)    



    if barcode_info[i][3]==0:
        colors_point.append("#000000")
    else:
        colors_point.append("#377eb8")
    #colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)

from pyvis.network import Network
import networkx as nx

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    
    label_str =  str(i)+'_c:'+str(barcode_info[i][3])+'_'
    if barcode_type[barcode_info[i][0]] == 0: 
        label_str = label_str + 'mixed'
        marker_size = 'circle'
    elif barcode_type[barcode_info[i][0]] == 1:
        label_str = label_str + 'Tcell'
        marker_size = 'box'
    elif barcode_type[barcode_info[i][0]] == 2:
        label_str = label_str + 'B'
        marker_size = 'circle'
    else:
        label_str = label_str + 'GC'
        marker_size = 'circle'

    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

threshold_value =  np.percentile(combined_score_distribution_ccl19_ccr7,0)
count_edges = 0
for k in range (1, len(csv_record)):
    if csv_record[k][4] < threshold_value:
        continue
    
    i = csv_record[k][6]
    j = csv_record[k][7]    
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    if ligand !='CCL19' or receptor != 'CCR7':
        continue
    
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record[k][4]
    g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score), color="#377eb8") 
    count_edges = count_edges + 1
    
nt.from_nx(g)
nt.show('mygraph.html')
 
        
 
'''
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):    
            #    #print('hello')
            #key_value_2 = lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            key_value = str(i) +'-'+ str(j) + '-' + lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            if len(csv_record_dict[key_value])>=1: #total_runs:  #key_value_2 in lr_target: #and : 
                edge_score =  min_attention_score + attention_scores[i][j][k]
                title_str =  "L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]#+", "+str(edge_score) #"L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]+", "+str(attention_scores[i][j][k])
                g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) #,width=, arrowsize=int(20),  arrowstyle='fancy'
				# label = title =
'''
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+"combined"+"_l1l2attention_allPairs_CCL19_CCR7_th80_80_0_"+str(count_edges)+"edges_5runs.dot")

write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_l1l2attention_allPairs_th80_80_90_"+str(count_edges)+"edges_5runs.dot")

write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_fullOutput_"+str(len(csv_record))+"edges.dot")
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_allPairs_th90_"+str(len(csv_record))+"edges.dot")
##################################### LUAD ##########################################################################################################

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

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    marker_size = 'circle'
    label_str =  str(i)+'_c:'+str(barcode_info[i][3])
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True
#################################
threshold_value =  np.percentile(combined_score_distribution,90)
count_edges = 0
for k in range (1, len(csv_record)):
    if csv_record[k][4] < threshold_value:
        continue
    
    i = csv_record[k][6]
    j = csv_record[k][7]    
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record[k][4]
    g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) 
    count_edges = count_edges + 1
######################################################    
'''
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):    
            #    #print('hello')
            #key_value_2 = lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            key_value = str(i) +'-'+ str(j) + '-' + lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            if len(csv_record_dict[key_value])>=1: #total_runs:  #key_value_2 in lr_target: #and : 
                edge_score =  min_attention_score + attention_scores[i][j][k]
                title_str =  "L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]#+", "+str(edge_score) #"L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]+", "+str(attention_scores[i][j][k])
                g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) #,width=, arrowsize=int(20),  arrowstyle='fancy'
				# label = title =
'''
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_l1l2attention_allPairs_th90_90_90_"+str(count_edges)+"edges.dot")

write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_th99p7_"+str(len(csv_record))+"edges_withScore.dot")
########################################### PDAC #############################################################################################

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
    y_index.append(-barcode_info[i][2])    
    colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)

from pyvis.network import Network
import networkx as nx

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    marker_size = 'circle'
    label_str =  str(i)+'_c:'+str(barcode_info[i][3])+'_'+str(barcode_type[barcode_info[i][0]])
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):    
            #    #print('hello')
            #key_value_2 = lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            key_value = str(i) +'-'+ str(j) + '-' + lig_rec_dict[i][j][k][0] + '-' + lig_rec_dict[i][j][k][1]
            if len(csv_record_dict[key_value])>=1: #total_runs:  #key_value_2 in lr_target: #and : 
                edge_score =  min_attention_score + attention_scores[i][j][k]
                title_str =  "L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]#+", "+str(edge_score) #"L:"+lig_rec_dict[i][j][k][0]+", R:"+lig_rec_dict[i][j][k][1]+", "+str(attention_scores[i][j][k])
                g.add_edge(int(i), int(j), label = title_str, value=np.float64(edge_score)) #,width=, arrowsize=int(20),  arrowstyle='fancy'
				# label = title =
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/interactive_"+args.data_name+"_"+filename[run_time]+"_th99p8_"+str(len(csv_record))+"edges.dot")



