import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc

barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
barcode_info.append("")
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append(line[0])

##############################################
emb_dim= 64 #512
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_melow/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/lambdaI0.3_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_mehigh/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/lambdaI0.8_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_r2/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_equal/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1_mp05/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1_mp2/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1mp5_r3/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1mp5_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/r6_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1mp5_temp/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/r1_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/norm_scale_pca_300fixed_lp8mp2_4layer_emb256_r7_Embed_X.npy' # r4 #test_r1_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_norm_scale_pca_2k_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/norm_scale_pca_2k_l1mp2_1layer_emb4096_r1_Embed_X.npy'
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/test_r4_Embed_X.npy' # r4
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/SAGEConv_r1_Embed_X.npy' # r4
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/RGCN_r1_Embed_X.npy' # r4
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/ChebConv_r1_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/GraphConv_r3_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/rgcn_r1_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/rgat_r1_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/rgat_r1_alongcell_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/gat_r1_2attr_withfeature_onlyccc_bidir_97_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/gat_r1_2attr_nofeature_onlyccc_bidir_97_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/gat_r1_2attr_nofeature_onlyccc_bidir_97_Embed_X.npy' # lr e-5, epoch 20k, K=3
#X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/gat_r1_2attr_withfeature_onlyccc_97_Embed_X.npy' # lr e-5, epoch 20k, K=3
X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/gat_r1_2attr_withAllFeature_bidir_Embed_X.npy' # lr e-5, epoch 20k, K=3


X_embedding = np.load(X_embedding_filename)
num_feature = X_embedding.shape[1]

#X_embedding= sc.pp.pca(X_embedding, n_comps=100) 
#num_feature = 100


#####################
feature_id = np.zeros((1, num_feature))
for i in range (0, num_feature):
    feature_id[0,i] = i+1

X_embedding = np.concatenate((feature_id, X_embedding))

X_embedding_T = np.transpose(X_embedding)
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_equal.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_l1_mp05.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_l1_mp2.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_melow.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_mehigh.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_bulk.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_l1mp5_r3.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/PCA_'+str(emb_dim)+'Embed_X_l1mp5_temp.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/GCN_r4.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/GCN_r7.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/TAGonv_test_r4_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/SAGEConv_r1_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/RGCN_r1_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/ChebConv_r1_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/GraphConv_r3_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/rgcn_r1_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/rgat_r1_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/rgat_r1_alongcell_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/rgat_r1_alongcell_normalized_ccc_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/gat_r1_2attr_withfeature_onlyccc_bidir_97_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/gat_r1_2attr_nofeature_onlyccc_bidir_97_node_embedding.csv'
#X_embedding_filename = '/cluster/home/t116508uhn/64630/gat_r1_2attr_withfeature_onlyccc_97_node_embedding.csv'
X_embedding_filename = '/cluster/home/t116508uhn/64630/gat_r1_2attr_withAllFeature_onlyccc_bidir_97_node_embedding.csv'
f=open(X_embedding_filename, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
# write the header
writer.writerow(barcode_info)
writer.writerows(X_embedding_T)
f.close()
####################################################

        
X_gene_data_path = '/cluster/projects/schwartzgroup/fatema/CCST/generated_data_noPCA_QuantileTransform/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/'
fp = open(X_gene_data_path + 'features', 'rb')
X_gene_data_2 = pickle.load(fp)
fp.close()
#X_gene_data = np.load(X_gene_data_path+'features.npy')
X_gene_data=scipy.sparse.csr_matrix(X_gene_data)
X_gene_data_T = np.transpose(X_gene_data)


X_gene_filename = '/cluster/home/t116508uhn/64630/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_X_gene.csv'
f=open(X_gene_filename, 'w', encoding='UTF8', newline='') #'/cluster/home/t116508uhn/test.csv'
writer = csv.writer(f)
# write the header
writer.writerow(barcode_info)
writer.writerows(X_gene_data_T)
f.close()

####################################################
emb_dim=32
#X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
#X_embedding_filename_2 = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_512_quantile/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/lambdaI1.0_Embed_X.npy'
X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_'+str(emb_dim)+'_quantile_weighted_TDistance_2k_l1mp5/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/Embed_X.npy'
X_embedding = np.load(X_embedding_filename)
X_embedding_T = np.transpose(X_embedding)

X_embedding_filename = '/cluster/home/t116508uhn/64630/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_'+str(emb_dim)+'Embed_X_l1mp5.csv'
f=open(X_embedding_filename, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
# write the header
writer.writerow(barcode_info)
writer.writerows(X_embedding_T)
f.close()

##############################################

cluster_label_file='/cluster/projects/schwartzgroup/fatema/CCST/result_NoPCA_512_quantile_weighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/cluster_label.csv'
cluster_label_info=[]
with open(cluster_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        cluster_label_info.append(line)

node_label=[]
for i in range (0, len(cluster_label_info)):
    node_label.append([barcode_info[i],int(cluster_label_info[i][1])])
    
node_label_file='/cluster/home/t116508uhn/64630/node_label.csv'
f=open(node_label_file, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
# write the header
writer.writerow(['item','label'])
writer.writerows(node_label)
f.close()

###############################################
barcode_label_file='/cluster/home/t116508uhn/64630/node_label.csv'
barcode_label_info=[]
with open(barcode_label_file) as file:
    tsv_file = csv.reader(file, delimiter=",")
    for line in tsv_file:
        barcode_label_info.append(line)
        
X_embedding_filename='/cluster/home/t116508uhn/64630/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_Embed_X.csv'
emb_label_info=[]
with open(X_embedding_filename) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        emb_label_info.append(line)
     
