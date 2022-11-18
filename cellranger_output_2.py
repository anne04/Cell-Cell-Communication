import os
#import glob
import pandas as pd
#import shutil
import csv
import numpy as np
import sys
import scikit_posthocs as post
import altair as alt
from collections import defaultdict
import stlearn as st
import scanpy as sc
import qnorm
import scipy
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()
th_dist = 4
k_nn = 4
spot_diameter = 89.43 #pixels


	
data_fold = args.data_path + 'filtered_feature_bc_matrix.h5'
print(data_fold)

#cell_vs_gene = adata_X   # rows = cells, columns = genes

datapoint_size = 2000
x_max = 500
x_min = 0
y_max = 300
y_min = 0
#################################

a = x_min
b = x_max
#coord_x = np.random.randint(a, b, size=(datapoint_size))
coord_x = (b - a) * np.random.random_sample(size=datapoint_size//2) + a

a = y_min
b = y_max
coord_y = (b - a) * np.random.random_sample(size=datapoint_size//2) + a
#coord_y = np.random.randint(a, b, size=(datapoint_size))

temp_x = coord_x
temp_y = coord_y


coord_x_t = np.random.normal(loc=100,scale=20,size=datapoint_size//8)
coord_y_t = np.random.normal(loc=100,scale=20,size=datapoint_size//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))

coord_x_t = np.random.normal(loc=400,scale=20,size=datapoint_size//8)
coord_y_t = np.random.normal(loc=100,scale=20,size=datapoint_size//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))


coord_x_t = np.random.normal(loc=200,scale=10,size=datapoint_size//8)
coord_y_t = np.random.normal(loc=150,scale=10,size=datapoint_size//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))
 
    
coord_x_t = np.random.normal(loc=400,scale=20,size=datapoint_size//8)
coord_y_t = np.random.normal(loc=200,scale=20,size=datapoint_size//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))


discard_points = dict()
for i in range (0, temp_x.shape[0]):
    if i not in discard_points:
        for j in range (i+1, temp_x.shape[0]):
            if j not in discard_points:
                if euclidean_distances(np.array([[temp_x[i],temp_y[i]]]), np.array([[temp_x[j],temp_y[j]]]))[0][0] < 1 :
                    print('i: %d and j: %d'%(i,j))
                    discard_points[j]=''

coord_x = []
coord_y = []
for i in range (0, temp_x.shape[0]):
    if i not in discard_points:
        coord_x.append(temp_x[i])
        coord_y.append(temp_y[i])

temp_x = np.array(coord_x)
temp_y = np.array(coord_y)

print(len(temp_x))
plt.scatter(x=np.array(temp_x), y=np.array(temp_y),s=1)

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'synthetic_spatial_plot_3.svg', dpi=400)
plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_uniform_normal', 'wb') as fp:
    pickle.dump([temp_x, temp_y], fp)

datapoint_size = temp_x.shape[0]
coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
distance_matrix = euclidean_distances(coordinates, coordinates)

dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))

for j in range(0, distance_matrix.shape[1]):
    max_value=np.max(distance_matrix[:,j])
    min_value=np.min(distance_matrix[:,j])
    for i in range(distance_matrix.shape[0]):
        dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)
	
    list_indx = list(np.argsort(dist_X[:,j]))
    k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
    for i in range(0, distance_matrix.shape[0]):
        if i not in k_higher:
            dist_X[i,j] = -1
	
	
'''distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
for i in range (0, datapoint_size):
    #ccc_j = []
    for j in range (0, datapoint_size):
        if distance_matrix[i][j]<th_dist:
            distance_matrix_threshold_I[i][j] = 1
            
distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I)
distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
with open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'total_synthetic_1_adjacency_matrix', 'wb') as fp:
    pickle.dump(distance_matrix_threshold_I_N_crs, fp)'''
    

region_list =  [[182, 230, 125, 170], [350, 450, 50, 150], [350, 450, 170, 225]]
ccc_scores_count = []
for region in region_list:
    count = 0
    for i in range (0, distance_matrix.shape[0]):
    #ccc_j = []
        for j in range (0, distance_matrix.shape[1]):
            if dist_X[i,j] > -1:            
                region_x_min = region[0]
                region_x_max = region[1]
                region_y_min = region[2]
                region_y_max = region[3]  		
                if temp_x[i] > region_x_min and temp_x[i] < region_x_max and temp_y[i] > region_y_min and temp_y[i] <  region_y_max: 
                    count = count + 1
    ccc_scores_count.append(count)          

		
a = 20
b = +558
limit_list =[[200,500],[20,50],[20,50]]
ccc_index_dict = dict()
row_col = []
edge_weight = []
for region_index in range (0, len(region_list)):
    region = region_list[region_index]
    a = limit_list[region_index][0]
    b = limit_list[region_index][1]
    ccc_scores = (b - a) * np.random.random_sample(size=ccc_scores_count[region_index]+1) + a
    k=0
    for i in range (0, distance_matrix.shape[0]):
        for j in range (0, distance_matrix.shape[1]):
            if dist_X[i,j] > -1:
                flag = 0          
                region_x_min = region[0]
                region_x_max = region[1]
                region_y_min = region[2]
                region_y_max = region[3]  		
                if temp_x[i] > region_x_min and temp_x[i] < region_x_max and temp_y[i] > region_y_min and temp_y[i] <  region_y_max: 
                    mean_ccc = ccc_scores[k]
                    k = k + 1
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    #edge_weight.append([0.5, mean_ccc])
                    #print([0.5, mean_ccc])
                    flag = 1
		    
print("len row_col with ccc %d"%len(row_col))

for i in range (0, distance_matrix.shape[0]):
    for j in range (0, distance_matrix.shape[1]):
        if dist_X[i,j] > -1:
            if i not in ccc_index_dict and j not in ccc_index_dict:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                #edge_weight.append([0.5, 0])



with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_uniform_normal', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)

print(len(row_col))
print(len(temp_x))

###############################################Visualization starts###################################################################################################

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_uniform_normal', 'rb') as fp:
    temp_x, temp_y = pickle.load(fp)

datapoint_size = temp_x.shape[0]

coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)


'''for i in range (0, distance_matrix.shape[0]):
    if np.sort(distance_matrix[i])[1]<0.1:
        print(np.sort(distance_matrix[i])[0:5])'''

#####################################
'''with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_uniform_normal', 'rb') as fp:             
    row_col, edge_weight = pickle.load(fp)


attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    attention_scores[i][j] = edge_weight[index][1]
    distribution.append(attention_scores[i][j])
    if edge_weight[index][1]>=0:
        ccc_index_dict[i] = ''
        ccc_index_dict[j] = ''    
'''
################

########
'''X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_region1_uniform_normal_knn_attention.npy'
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = np.zeros((temp_x.shape[0],temp_x.shape[0]))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[1][index][0]
    distribution.append(attention_scores[i][j])
	'''
###############
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r2_2attr_noFeature_STnCCC_region1_uniform_normal_knn_attention.npy'
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = np.zeros((temp_x.shape[0],temp_x.shape[0]))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[2][index][0]
    distribution.append(attention_scores[i][j])
##############
attention_scores_normalized = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores_normalized [i][j] = X_attention_bundle[1][index][0]
##############
adjacency_matrix = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    adjacency_matrix [i][j] = 1


##############


ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 90)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] >= threshold_down and attention_scores[i][j] <= threshold_up: #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            '''ccc_index_dict[i] = ''
            ccc_index_dict[j] = '' 
	    '''
            
############
'''region_list =  [[182, 230, 125, 170], [350, 450, 50, 225], [70, 120, 70, 125]]
percentile_value = 85
threshold =  np.percentile(sorted(distribution), percentile_value)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        flag = 0
        for region in region_list:
            region_x_min = region[0]
            region_x_max = region[1]
            region_y_min = region[2]
            region_y_max = region[3]  		
            if temp_x[i] > region_x_min and temp_x[i] < region_x_max and temp_y[i] > region_y_min and temp_y[i] <  region_y_max:                				
                if attention_scores[i][j] > threshold and attention_scores[i][j] < np.percentile(sorted(distribution), 100): #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1'''
    
#############


graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
print('number of component %d'%n_components)

count_points_component = np.zeros((n_components))
for i in range (0, len(labels)):
     count_points_component[labels[i]] = count_points_component[labels[i]] + 1
           
print(count_points_component)

id_label = 0  
index_dict = dict()
for i in range (0, count_points_component.shape[0]):
    if count_points_component[i]>1:
        id_label = id_label+1
        index_dict[i] = id_label
print(id_label)

datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if count_points_component[labels[i]]>1:
        datapoint_label.append(2) #
        #datapoint_label.append(index_dict[labels[i]])
    else:
        datapoint_label.append(0)
	
#############
'''datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if i in ccc_index_dict:
        datapoint_label.append(1)
    else:
        datapoint_label.append(0)
id_label=2
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

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

       
#exist_datapoint = dict()
id_label = [0,2]
for j in id_label:
#for j in range (0, id_label+1):
    x_index=[]
    y_index=[]
    #fillstyles_type = []
    for i in range (0, temp_x.shape[0]):
        if datapoint_label[i] == j:
            x_index.append(temp_x[i])
            y_index.append(temp_y[i])
    print(len(x_index))
            
            ##############
    plt.scatter(x=x_index, y=y_index, label=j, color=colors[j], s=1)   
    
plt.legend(fontsize=4,loc='upper right')


save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 




from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
gene_info=dict()
#barcode_info.append("")
i=0
with open(gene_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        gene_info[line[1]]=''

ligand_dict_dataset = defaultdict(list)


ligand_dict_db = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'

'''df = pd.read_csv(cell_chat_file)
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        ligand_dict_db[ligand].append(receptor)'''

df = pd.read_csv(cell_chat_file)
cell_cell_contact = []
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    if ligand not in gene_info:
        continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
            ligand_dict_dataset[ligand].append(receptor)
            #######
            if df["annotation"][i] == 'Cell-Cell Contact':
                cell_cell_contact.append(receptor)
            #######                
            
print(len(ligand_dict_dataset.keys()))

nichetalk_file = '/cluster/home/t116508uhn/64630/NicheNet-LR-pairs.csv'   
df = pd.read_csv(nichetalk_file)
for i in range (0, df["from"].shape[0]):
    ligand = df["from"][i]
    if ligand not in gene_info:
        continue
    receptor = df["to"][i]
    ligand_dict_dataset[ligand].append(receptor)
    
print(len(ligand_dict_dataset.keys()))

##################################################################
total_relation = 0
l_r_pair = dict()
count = 0
for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    l_r_pair[gene] = dict()
    for receptor_gene in ligand_dict_dataset[gene]:
        l_r_pair[gene][receptor_gene] = -1 #count #
        count = count + 1
##################################################################
print(count)
##################################################################

#####################

cell_vs_gene_dict = []
gene_list = defaultdict(list)
#all_expression = []
for cell_index in range (0, cell_vs_gene.shape[0]):
    cell_vs_gene_dict.append(dict())
    gene_exp = cell_vs_gene[cell_index]
    for gene_i in range (0, len(gene_exp)):
        gene_list[gene_ids[gene_i]].append(gene_exp[gene_i])
        cell_vs_gene_dict[cell_index][gene_ids[gene_i]] = gene_exp[gene_i]
        #all_expression.append(gene_exp[gene_i])
        
##########
'''i = 0
for gene in gene_ids:
    df = pd.DataFrame (gene_list[gene], columns = ['gene_expression'])
    chart = alt.Chart(df).transform_density(
        'gene_expression',
        as_=['gene_expression', 'density'],
    ).mark_area().encode(
        x="gene_expression:Q",
        y='density:Q',
    )
    save_path = '/cluster/home/t116508uhn/64630/'
    chart.save(save_path+'gene_exp_dist_'+gene+'.svg')
    print(i)
    i = i+1'''
##########        
        
gene_list_percentile = defaultdict(list)
for gene in gene_ids:
    gene_list_percentile[gene].append(np.percentile(sorted(gene_list[gene]), 70))
    gene_list_percentile[gene].append(np.percentile(sorted(gene_list[gene]), 97))   

#global_percentile = np.percentile(all_expression, 99)
#################################################
cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
    cell_percentile.append([np.percentile(sorted(cell_vs_gene[i]), 70),np.percentile(sorted(cell_vs_gene[i]), 80), np.percentile(sorted(cell_vs_gene[i]), 97)])
##################################################


cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
for i in range (0, cell_vs_gene.shape[0]): # ligand
    count_rec = 0
    #max_score = -1
    #min_score = 1000
    
    
	
    for gene in list(ligand_dict_dataset.keys()): 
        if (gene in gene_list) and cell_vs_gene_dict[i][gene] >= cell_percentile[i][2]: # gene_list_percentile[gene][1]: #global_percentile: #
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                for gene_rec in ligand_dict_dataset[gene]:
                    if (gene_rec in gene_list) and cell_vs_gene_dict[j][gene_rec] >= cell_percentile[j][2]: #gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                            continue
                        else:
                            if distance_matrix[i,j] > spot_diameter*4:
                                continue
                            communication_score = cell_vs_gene_dict[i][gene] * cell_vs_gene_dict[j][gene_rec]
                            
                            '''if communication_score > max_score:
                                max_score = communication_score
                            if communication_score < min_score:
                                min_score = communication_score ''' 
                                
                            if l_r_pair[gene][gene_rec] == -1: 
                                l_r_pair[gene][gene_rec] = pair_id
                                pair_id = pair_id + 1 
                           
                            relation_id = l_r_pair[gene][gene_rec]
                            cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
                            count_rec = count_rec + 1
                            count_total_edges = count_total_edges + 1
                            
                            
    cell_rec_count[i] =  count_rec   
    print("%d - %d "%(i, count_rec))
    #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))
    
print(pair_id)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'ligand-receptor-records_97', 'wb') as fp:
    pickle.dump([cells_ligand_vs_receptor,ligand_dict_dataset,pair_id, cell_rec_count], fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'ligand-receptor-records_97', 'rb') as fp:
    cells_ligand_vs_receptor, ligand_dict_dataset, pair_id, cell_rec_count = pickle.load(fp)

i = 0
for j in range (0, len(cells_ligand_vs_receptor)):
    if cells_ligand_vs_receptor[i][j]>0:
	print(cells_ligand_vs_receptor[i][j])
'''row_col = []
edge_weight = []
edge_type = []
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        if distance_matrix[i][j]<300:
            row_col.append([i,j])
            if i==j: 
                edge_weight.append(0.8)
            else:
                edge_weight.append(0.2)
            edge_type.append(0)  
            
            if len(cells_ligand_vs_receptor[i][j])>0:  
		
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):  
                    row_col.append([i,j])
                    edge_weight.append(cells_ligand_vs_receptor[i][j][k][2])
                    edge_type.append(cells_ligand_vs_receptor[i][j][k][3])  '''
            

row_col = []
edge_weight = []
edge_type = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            row_col.append([i,j])
            if i==j: 
                edge_weight.append(0.5)
            else:
                edge_weight.append(0.5)
            edge_type.append(0)  
            
            if len(cells_ligand_vs_receptor[i][j])>0:
				mean_ccc = 0
				for k in range (0, len(cells_ligand_vs_receptor[i][j])):
					mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
				mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
				row_col.append([i,j])
				edge_weight.append(mean_ccc)
				edge_type.append(1) 

				
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records', 'wb') as fp:
    pickle.dump([row_col, edge_weight, edge_type], fp)				
				
row_col = []
edge_weight = []
edge_type = []
for i in range (0, len(cells_ligand_vs_receptor)):
    ccc_j = []
	ccc_score = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            row_col.append([i,j])
            if i==j: 
                edge_weight.append(0.5)
            else:
                edge_weight.append(0.5)
            edge_type.append(0)  
            
            if len(cells_ligand_vs_receptor[i][j])>0:
				mean_ccc = 0
				for k in range (0, len(cells_ligand_vs_receptor[i][j])):
					mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
				mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
				
				ccc_score.append(mean_ccc)
				ccc_j.append(j)
				
	sum_score = np.sum(ccc_score)			
	for j in range (0, len(ccc_j)):
		row_col.append([i,ccc_j[j]])
		edge_weight.append(ccc_score[j]/sum_score)
		edge_type.append(1) 

				
print(len(row_col))				
				
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_normalized_ccc', 'wb') as fp:
    pickle.dump([row_col, edge_weight, edge_type], fp)
           
            
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            row_col.append([i,j])
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                edge_weight.append([0.5, mean_ccc])
            else:
                edge_weight.append([0.5, 0])
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
	  

##############################
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                row_col.append([i,j])
                edge_weight.append([0.5, mean_ccc])
            elif i==j: # if not onlyccc, then remove the condition i==j
                row_col.append([i,j])
                edge_weight.append([0.5, 0])
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
	  

##############################


edge_weight_temp = []
for i in range (0, len(cells_ligand_vs_receptor)):
    edge_weight_temp.append([])
    
for i in range (0, len(cells_ligand_vs_receptor)):
    for j in range (0, len(cells_ligand_vs_receptor)):
        edge_weight_temp[i].append([])
        edge_weight_temp[i][j] = []


for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                edge_weight_temp[i][j].append(0.5)
                edge_weight_temp[i][j].append(mean_ccc)  
            elif i==j : # required for self knowledge. Do it for i!=j as well if for link prediction ### SEE THIS ###
                edge_weight_temp[i][j].append(0.5)			
                edge_weight_temp[i][j].append(0) 
				
row_col = []
edge_weight = []				
for i in range (0, len(cells_ligand_vs_receptor)):
	for j in range (i, len(cells_ligand_vs_receptor)):
		if i==j: 
			edge_weight_temp[i][j].append(0) # make it length 3
			temp_weight = edge_weight_temp[i][j]
			row_col.append([i,j])
			edge_weight.append(temp_weight)
		elif len(edge_weight_temp[i][j])>0:
			temp_weight = edge_weight_temp[i][j] + edge_weight_temp[j][i]
			if len(temp_weight) == 2: 
				temp_weight.append(0)
			elif len(temp_weight) == 4:
				temp_weight = [temp_weight[0], temp_weight[1], temp_weight[3]]
			#else: # len = 0 -- don't add the edge 
			
			row_col.append([i,j])
			edge_weight.append(temp_weight)
			row_col.append([j,i])
			edge_weight.append(temp_weight)

             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_onlyccc_bidir_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
	  
fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_onlyccc_70', 'rb')
row_col, edge_weight = pickle.load(fp)

i=0
for tupple in row_col:
	if tupple[1] == 192: 
		print(tupple)
		print(edge_weight[i])
	i=i+1
	
'''for i in range (0, cell_vs_gene.shape[0]): 
    for j in range (0, cell_vs_gene.shape[0]): 
        
        if len(cells_ligand_vs_receptor[i][j]) != 0:
            print(j)
            print(cells_ligand_vs_receptor[i][j])'''
        
        





    
            
            
