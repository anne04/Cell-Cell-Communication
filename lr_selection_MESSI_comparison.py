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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()



spot_diameter = 89.43 #pixels
############

############
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
 
####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
print(data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
#################### 
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
#adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X   # rows = cells, columns = genes

####################
'''
adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
#adata_X = sc.pp.scale(adata_X)
#adata_X = sc.pp.pca(adata_X, n_comps=args.Dim_PCA)
cell_vs_gene = sparse.csr_matrix.toarray(adata_X) #adata_X
'''
####################
####################
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    cell_percentile.append([np.percentile(sorted(cell_vs_gene[i]), 5), np.percentile(sorted(cell_vs_gene[i]), 50),np.percentile(sorted(cell_vs_gene[i]), 70), np.percentile(sorted(cell_vs_gene[i]), 97)])

#gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406


gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
	

'''lr_pairs = pd.read_html('/cluster/home/t116508uhn/MESSI/input/ligand_receptor_pairs2.txt', header=None)[0] #pd.read_table(os.path.join(input_path, filename), header=None)
lr_pairs.columns = ['ligand','receptor']
lr_pairs[['ligand','receptor']] = lr_pairs['receptor'].str.split('\t',expand=True)
lr_pairs['ligand'] = lr_pairs['ligand'].apply(lambda x: x.upper())
lr_pairs['receptor'] = lr_pairs['receptor'].apply(lambda x: x.upper())


selected_receptor_list = ['CCKBR', 'GALR1', 'AVPR1A', 'OXTR', 'OPRD1']
ligand_dict_dataset = defaultdict(list)
for index in range (0, len(lr_pairs)):
    if lr_pairs['receptor'][index] in selected_receptor_list and lr_pairs['receptor'][index] in gene_index and lr_pairs['ligand'][index] in gene_index:
        ligand_dict_dataset[lr_pairs['ligand'][index]].append(lr_pairs['receptor'][index])
   
print(len(ligand_dict_dataset.keys()))
'''

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'MESSI_predicted_lr_pairs', 'rb') as fp: 
    ligand_dict_dataset = pickle.load(fp) 

for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    gene_info[gene] = 'included'
    for receptor_gene in ligand_dict_dataset[gene]:
        gene_info[receptor_gene] = 'included'
   
count = 0
for gene in gene_info.keys(): 
    if gene_info[gene] == 'included':
        count = count + 1
print(count)
affected_gene_count = count


######################################
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

ligand_list = list(ligand_dict_dataset.keys())  

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
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
            dist_X[i,j] = -1
            
cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
activated_cell_index = dict()
#count = 0
#count2 = 0
print('len ligand_list %d'%len(ligand_list))

start_index = args.slice
end_index = min(len(ligand_list), start_index+100)

for gene in ligand_list: #[0:5]: 
    for i in range (0, cell_vs_gene.shape[0]): # ligand
        count_rec = 0    
        if cell_vs_gene[i][gene_index[gene]] > 0:
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] > 0: #gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if distance_matrix[i,j] <= spot_diameter*4:                                                   
                            communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                            '''if gene=='L1CAM':
                                count = count+1
                            elif gene=='LAMC2':
                                count2 = count2+1'''
                            if l_r_pair[gene][gene_rec] == -1: 
                                l_r_pair[gene][gene_rec] = pair_id
                                pair_id = pair_id + 1 
                           
                            relation_id = l_r_pair[gene][gene_rec]
                            cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
                            count_rec = count_rec + 1
                            count_total_edges = count_total_edges + 1
                            activated_cell_index[i] = ''
                            activated_cell_index[j] = ''
                            
                            
        cell_rec_count[i] =  count_rec   
        #print("%d - %d "%(i, count_rec))
        #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))
    
    print(pair_id)
	
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_selective_lr_STnCCC_MESSI_comparison_a', 'wb') as fp: #b, b_1, a
#    pickle.dump([cells_ligand_vs_receptor,l_r_pair,ligand_list,activated_cell_index], fp) #a - [0:5]


ccc_index_dict = dict()
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4:
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                row_col.append([i,j])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                edge_weight.append([dist_X[i,j], mean_ccc])
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])

		
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnAvgCCC_MESSI_comparison_a', 'wb') as fp:  #b, a:[0:5]           
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_MESSI_predicted_pairs', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
############################################################

###########################################################
pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if pathologist_label[i][1] == 'tumor': #'Tumour':
        barcode_type[pathologist_label[i][0]] = 1
    elif pathologist_label[i][1] =='stroma_deserted':
        barcode_type[pathologist_label[i][0]] = 0
    elif pathologist_label[i][1] =='acinar_reactive':
        barcode_type[pathologist_label[i][0]] = 2
    else:
        barcode_type[pathologist_label[i][0]] = 0


coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
        
##########
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_selective_lr_STnCCC_MESSI_comparison_a', 'rb') as fp: #b, b_1, a
    cells_ligand_vs_receptor,l_r_pair,ligand_list,activated_cell_index = pickle.load(fp) 

datapoint_size = len(cells_ligand_vs_receptor)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_MESSI_predicted_pairs', 'rb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+'all_avg', 'rb') as fp:  #b, a:[0:5]           		
    row_col, edge_weight = pickle.load(fp)

attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    
    attention_scores[i][j] = edge_weight[index][1]
    if edge_weight[index][1]>0:
        distribution.append(attention_scores[i][j])
        ccc_index_dict[i] = ''
        #ccc_index_dict[j] = ''    

###########
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnCCC_c_70_attention.npy' #a
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnAvgCCC_MESSI_a_bothlayer_attention_l1.npy' #a
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[3][index][0] #X_attention_bundle[2][index][0]
    #if attention_scores[i][j]>0:
    distribution.append(attention_scores[i][j])
##############
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


##############


ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 96)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] >= threshold_down and attention_scores[i][j] <= threshold_up: #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            ccc_index_dict[i] = ''
	

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
    
 
for i in range (0, len(barcode_info)):
#    if barcode_info[i][0] in barcode_label:
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    else: 
        barcode_info[i][3] = 0
       

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
#######
'''for i in range (0, len(barcode_info)):
    if i in ccc_index_dict:
        barcode_info[i][3] = 2
    else: 
        barcode_info[i][3] = 0
'''
#######
'''id_label = [0,2]
for j in id_label:
'''
for j in range (0, n_components):
    label_i = j
    x_index=[]
    y_index=[]
    marker_size = []
    #fillstyles_type = []
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == j:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            #cell_count_cluster[j] = cell_count_cluster[j]+1
            spot_color = colors[j]
            if barcode_type[barcode_info[i][0]] == 0:
                marker_size.append('o') 
                #fillstyles_type.append('full') 
            elif barcode_type[barcode_info[i][0]] == 1:
                marker_size.append('^')  
                #fillstyles_type.append('full') 
            else:
                marker_size.append('*') 
                #fillstyles_type.append('full') 
            
            ###############
            
    
    for i in range (0, len(x_index)):  
        plt.scatter(x=x_index[i], y=-y_index[i], label = j, color=colors[j], marker=matplotlib.markers.MarkerStyle(marker=marker_size[i], fillstyle=filltype), s=15)   
    filltype = 'full'
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=spot_color, marker=marker_size)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    
#plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 
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

    
g = nx.DiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    if barcode_type[barcode_info[i][0]] == 0:
        marker_size = 'circle'
    elif barcode_type[barcode_info[i][0]] == 1:
        marker_size = 'box'
    else:
        marker_size = 'ellipse'
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network("500px", "500px", directed=True)
nt.from_nx(g)
for i in range (0, attention_scores.shape[0]):
    for j in range (0, attention_scores.shape[1]):
        if attention_scores[i][j] >= threshold_down:
            #print('hello')
            nt.add_edge(int(i), int(j)) #, weight=1, arrowsize=int(20),  arrowstyle='fancy'

nt.show('mygraph.html')


#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html


g = nx.Graph()
#g = Network("500px", "500px", directed=True)
g.add_node(1, x=1, y=1, title='I am node 1', label='NODE 1', color='#00ff1e')
g.add_node(2, x=10, y=2, title='I am node 2', label='NODE 2', color='#162347')
g.add_node(3, x=1, y=3, title='I am node 3', label='NODE 3', color='#dd4b39')

nt = Network('500px', '500px')
nt.from_nx(g)
nt.show('mygraph.html')

#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html

g.add_nodes([1,2,3],
    ...:                          title=['I am node 1', 'node 2 here', 'and im n
    ...: ode 3'],
    ...:                          x=[1, 1, 1],
    ...:                          y=[1, 2, 3],
    ...:                          label=['NODE 1', 'NODE 2', 'NODE 3'],
    ...:                          color=['#00ff1e', '#162347', '#dd4b39'])

import numpy as np
import matplotlib.pyplot as plt

					 
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
