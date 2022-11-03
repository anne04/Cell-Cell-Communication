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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
args = parser.parse_args()

spot_diameter = 89.43 #pixels


	
data_fold = args.data_path + 'filtered_feature_bc_matrix.h5'
print(data_fold)
adata_h5 = sc.read_10x_h5(data_fold)
print(adata_h5)

sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)

gene_ids = list(adata_h5.var_names)

temp = qnorm.quantile_normalize(np.transpose(scipy.sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X   # rows = cells, columns = genes
#################################
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
'''barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1 '''
x_max = np.max(coordinates[:,0])
x_min = np.min(coordinates[:,0])
y_max = np.max(coordinates[:,1])
y_min = np.min(coordinates[:,1])
#################################

a = x_min-spot_diameter//2
b = x_max+spot_diameter//2
#coord_x = (b - a) * np.random.normal(size=adata_h5.X.shape[0]) + a
coord_x = (b - a) * np.random.random_sample(size=adata_h5.X.shape[0]//2) + a

a = y_min-spot_diameter//2
b = y_max+spot_diameter//2
#coord_y = (b - a) * np.random.normal(size=adata_h5.X.shape[0]) + a
coord_y = (b - a) * np.random.random_sample(size=adata_h5.X.shape[0]//2) + a

save_x=coord_x
save_y=coord_y
temp_x = coord_x
temp_y = coord_y


coord_x_t = np.random.normal(loc=5000,scale=300,size=adata_h5.X.shape[0]//8)
coord_y_t = np.random.normal(loc=7000,scale=300,size=adata_h5.X.shape[0]//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))

coord_x_t = np.random.normal(loc=5000,scale=300,size=adata_h5.X.shape[0]//8)
coord_y_t = np.random.normal(loc=11000,scale=300,size=adata_h5.X.shape[0]//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))

ccc_scores = np.random.random_sample(size=affected_gene_count)

ccc_dict_x = dict()
ccc_dict_y = dict()
for i in range (0, coord_x_t.shape[0]):
    ccc_dict_x[coord_x_t[i]] = ''
    ccc_dict_y[coord_y_t[i]] = ''


coord_x_t = np.random.normal(loc=8000,scale=900,size=adata_h5.X.shape[0]//8)
coord_y_t = np.random.normal(loc=9000,scale=900,size=adata_h5.X.shape[0]//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))

coord_x_t = np.random.normal(loc=10000,scale=1000,size=adata_h5.X.shape[0]//8)
coord_y_t = np.random.normal(loc=10000,scale=1000,size=adata_h5.X.shape[0]//8)
temp_x = np.concatenate((temp_x,coord_x_t))
temp_y = np.concatenate((temp_y,coord_y_t))

plt.scatter(x=np.array(temp_x), y=np.array(temp_y),s=1)
#plt.scatter(x=np.array(coord_x), y=-np.array(coord_y),s=1)

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'synthetic_spatial_plot_2.svg', dpi=400)
plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_2', 'wb') as fp:
    pickle.dump([temp_x, temp_y], fp)

from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

ccc_index_dict = dict()
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<3:
            #if i==j:
            if i in ccc_dict_x and j in ccc_dict_y:
                mean_ccc = np.random(1)
                row_col.append([i,j])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                edge_weight.append([0.5, mean_ccc])
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:
                row_col.append([i,j])
                edge_weight.append([0.5, 0])

		
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region2_STnCCC_70', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
########

attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
distribution = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    attention_scores[i][j] = edge_weight[index][1]
    distribution.append(attention_scores[i][j])
    

threshold =  np.percentile(sorted(distribution), 90)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))

for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] > threshold: #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            
############


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

    
cell_count_cluster=np.zeros((labels.shape[0]))
filltype='none'
for j in range (0, n_components):
    label_i = j
    x_index=[]
    y_index=[]
    marker_size = []
    #fillstyles_type = []
    for i in range (0, temp_x.shape[0]):
        if label[i] == j:
            x_index.append(temp_x[i])
            y_index.append(temp_y[i])
            
            '''if barcode_type[barcode_info[i][0]] == 0:
                marker_size.append('o') 
                #fillstyles_type.append('full') 
            elif barcode_type[barcode_info[i][0]] == 1:
                marker_size.append('^')  
                #fillstyles_type.append('full') 
            else:
                marker_size.append('*') 
                #fillstyles_type.append('full')''' 
            
            ###############
            
    
    for i in range (0, len(x_index)):  
        plt.scatter(x=x_index[i], y=-y_index[i], label = j, color=colors[j], s=1)   
    #filltype = 'full'
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=spot_color, marker=marker_size)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    
#plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
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
        
        





    
            
            
