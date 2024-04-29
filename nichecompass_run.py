import os
import random
import warnings
from datetime import datetime
#import gdown

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
import altair as alt
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                compute_communication_gp_network,
                                visualize_communication_gp_network,
                                create_new_color_dict,
                                extract_gp_dict_from_mebocost_es_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                generate_enriched_gp_info_plots)

### Dataset ###
#dataset = "starmap_plus_mouse_cns"
species = "human"
spatial_key = "spatial"
n_neighbors = 4

### Model ###
# AnnData Keys
counts_key = "counts"
adj_key = "spatial_connectivities"
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

# Architecture
conv_layer_encoder = "gcnconv" # change to "gatv2conv" if enough compute and memory
active_gp_thresh_ratio = 0.01

# Trainer
n_epochs = 400
n_epochs_all_gps = 25
lr = 0.01
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_l1_masked = 0. # increase if gene selection desired
lambda_l1_addon = 100.
edge_batch_size = 1024 # increase if more memory available
n_sampled_neighbors = 4
use_cuda_if_available = True

### Analysis ###
cell_type_key = "Main_molecular_cell_type"
latent_leiden_resolution = 0.01  
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 0.2
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

# Get time of notebook execution for timestamping saved artifacts
now = datetime.now()
current_timestamp = '25042024_171242' #now.strftime("%d%m%Y_%H%M%S")

# Define paths
ga_data_folder_path = "data/gene_annotations"
gp_data_folder_path = "data/gene_programs"
so_data_folder_path = "data/spatial_omics"

omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
collectri_tf_network_file_path = f"{gp_data_folder_path}/collectri_tf_network_{species}.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
nichenet_ligand_target_matrix_file_path = f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"

mebocost_enzyme_sensor_interactions_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
gene_orthologs_mapping_file_path = f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
artifacts_folder_path = f"artifacts"
model_folder_path = f"{artifacts_folder_path}/single_sample/{current_timestamp}/model"
figure_folder_path = f"{artifacts_folder_path}/single_sample/{current_timestamp}/figures"

os.makedirs(model_folder_path, exist_ok=True)
os.makedirs(figure_folder_path, exist_ok=True)
#os.makedirs(so_data_folder_path, exist_ok=True)

# Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    species=species,
    min_curation_effort=0,
    load_from_disk=True,
    save_to_disk=True,
    lr_network_file_path=omnipath_lr_network_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True,
    gp_gene_count_distributions_save_path=f"{figure_folder_path}" \
                                           "/omnipath_gp_gene_count_distributions.svg")

# Display example OmniPath GP
omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]
print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")

# Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)
mebocost_gp_dict = extract_gp_dict_from_mebocost_es_interactions(
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,
    species=species,
    plot_gp_gene_count_distributions=True)

# Display example MEBOCOST GP
mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]
print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")

# Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)
nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,
    version="v2",
    keep_target_genes_ratio=1.,
    max_n_target_genes_per_gp=250,
    load_from_disk=True,
    save_to_disk=True,
    lr_network_file_path=nichenet_lr_network_file_path,
    ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True)

# Display example NicheNet GP
nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]
print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")

# Add GPs into one combined dictionary for model training
combined_gp_dict = dict(omnipath_gp_dict)
combined_gp_dict.update(mebocost_gp_dict)
combined_gp_dict.update(nichenet_gp_dict)

# Filter and combine GPs to avoid overlaps
combined_new_gp_dict = filter_and_combine_gp_dict_gps(
    gp_dict=combined_gp_dict,
    gp_filter_mode="subset",
    combine_overlap_gps=True,
    overlap_thresh_source_genes=0.9,
    overlap_thresh_target_genes=0.9,
    overlap_thresh_genes=0.9)

print("Number of gene programs before filtering and combining: "
      f"{len(combined_gp_dict)}.")
print(f"Number of gene programs after filtering and combining: "
      f"{len(combined_new_gp_dict)}.")

# Read data
'''
adata = sc.read_h5ad(
        f"{so_data_folder_path}/{dataset}_batch1.h5ad")
'''
######### Human lymph #######################################
adata = sc.read_visium("/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/")
adata.layers['counts']=adata.X
adata_batch = []

for i in range (0, adata.X.shape[0]):
     adata_batch.append('sagittal1')

adata.obs['batch'] = adata_batch
################################################

# Compute spatial neighborhood
sq.gr.spatial_neighbors(adata,
                        coord_type="generic",
                        spatial_key=spatial_key,
                        n_neighs=n_neighbors)

# Make adjacency matrix symmetric
adata.obsp[adj_key] = (
    adata.obsp[adj_key].maximum(
        adata.obsp[adj_key].T))

# Add the GP dictionary as binary masks to the adata
add_gps_from_gp_dict_to_adata(
    gp_dict=combined_new_gp_dict,
    adata=adata,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=1,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None)
'''
cell_type_colors = create_new_color_dict(
    adata=adata,
    cat_key=cell_type_key)
'''
print(f"Number of nodes (observations): {adata.layers['counts'].shape[0]}")
print(f"Number of node features (genes): {adata.layers['counts'].shape[1]}")
'''
# Visualize cell-level annotated data in physical space
sc.pl.spatial(adata,
              color=cell_type_key,
              palette=cell_type_colors,
              spot_size=spot_size)        
'''
#load_timestamp = "09022024_180928"
load_timestamp = current_timestamp # uncomment if you trained the model in this notebook

figure_folder_path = f"{artifacts_folder_path}/single_sample/{load_timestamp}/figures"
model_folder_path = f"{artifacts_folder_path}/single_sample/{load_timestamp}/model"

os.makedirs(figure_folder_path, exist_ok=True)
# Train model
'''
# Initialize model
model = NicheCompass(adata,
                     counts_key=counts_key,
                     adj_key=adj_key,
                     gp_names_key=gp_names_key,
                     active_gp_names_key=active_gp_names_key,
                     gp_targets_mask_key=gp_targets_mask_key,
                     gp_targets_categories_mask_key=gp_targets_categories_mask_key,
                     gp_sources_mask_key=gp_sources_mask_key,
                     gp_sources_categories_mask_key=gp_sources_categories_mask_key,
                     latent_key=latent_key,
                     conv_layer_encoder=conv_layer_encoder,
                     active_gp_thresh_ratio=active_gp_thresh_ratio)

# Train model
model.train(n_epochs=n_epochs,
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_l1_masked=lambda_l1_masked,
            edge_batch_size=edge_batch_size,
            n_sampled_neighbors=n_sampled_neighbors,
            use_cuda_if_available=use_cuda_if_available,
            verbose=False)
# Compute latent neighbor graph
sc.pp.neighbors(model.adata,
                use_rep=latent_key,
                key_added=latent_key)

# Compute UMAP embedding
sc.tl.umap(model.adata,
           neighbors_key=latent_key)

# Save trained model
model.save(dir_path=model_folder_path,
           overwrite=True,
           save_adata=True,
           adata_file_name="adata.h5ad")
'''

###################################################################


# Load trained model
model = NicheCompass.load(dir_path=model_folder_path,
                          adata=None,
                          adata_file_name="adata.h5ad",
                          gp_names_key=gp_names_key)

# Compute UMAP embedding
sc.tl.umap(model.adata,
           neighbors_key=latent_key)

samples = model.adata.obs[sample_key].unique().tolist()
'''
cell_type_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=cell_type_key)
'''
# Compute latent Leiden clustering
sc.tl.leiden(adata=model.adata,
             resolution=latent_leiden_resolution,
             key_added=latent_cluster_key,
             neighbors_key=latent_key)

latent_cluster_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=latent_cluster_key)

# Create plot of latent cluster/niche annotations in physical and latent space
groups = None # set this to a specific cluster for easy visualization, e.g. ["17"]
save_fig = True
file_path = f"{figure_folder_path}/" \
            f"res_{latent_leiden_resolution}_" \
            "niches_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"NicheCompass Niches " \
                       "in Latent and Physical Space",
                     y=0.96,
                     x=0.55,
                     fontsize=20)
spec1 = gridspec.GridSpec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples),
                          nrows=2,
                          width_ratios=[1] * len(samples),
                          height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))

sc.pl.umap(adata=model.adata,
           color=[latent_cluster_key],
           groups=groups,
           palette=latent_cluster_colors,
           title=f"Niches in Latent Space",
           ax=axs[0],
           show=False)

for idx, sample in enumerate(samples):
    print('hei')
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[latent_cluster_key],
                  groups=groups,
                  palette=latent_cluster_colors,
                  spot_size=100,
                  title=f"Niches in Physical Space \n"
                        f"(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False,
                  save='tissue_plot.svg')

# Create and position shared legend
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles,
                 labels,
                 loc="center left",
                 bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

# Adjust, save and display plot
plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path,
                bbox_extra_artists=(lgd, title),
                bbox_inches="tight")
plt.show()
plt.savefig('/cluster/home/t116508uhn/'+'tissue_plot.svg', dpi=400)
plt.clf()
 
####################

# Check number of active GPs
active_gps = model.get_active_gps()
print(f"Number of total gene programs: {len(model.adata.uns[gp_names_key])}.")
print(f"Number of active gene programs: {len(active_gps)}.")

# Display example active GPs
gp_summary_df = model.get_gp_summary()
gp_summary_df[gp_summary_df["gp_active"] == True].head()
gp_summary_df_active = gp_summary_df[gp_summary_df["gp_active"] == True]
gp_summary_df_active.to_csv('lymph_gp_summary_df_active.csv')

source_gene_weight = []
for i in range (0, len(gp_summary_df_active)):
    index = gp_summary_df_active.index[i]
    if len(gp_summary_df_active["gp_source_genes"][index])>1:
        avg_importance = np.mean(gp_summary_df_active["gp_source_genes_importances"][index])
        source_gene_weight.append([gp_summary_df_active["gp_name"][index], avg_importance, gp_summary_df_active["gp_target_genes"][index], index])
    else:
        source_gene_weight.append([gp_summary_df_active["gp_source_genes"][index][0], gp_summary_df_active["gp_source_genes_importances"][index][0],gp_summary_df_active["gp_target_genes"][index], index])

source_gene_weight = sorted(source_gene_weight, key = lambda x: x[1], reverse=True)
for i in range (0, len(source_gene_weight)):
    if "CCR7" in source_gene_weight[i][2] and ("CCL19" in source_gene_weight[i][0] or "CCL21" in source_gene_weight[i][0]):
        print(i)
        
#
#CCL21 - CCR7 is found at position 32
#CCL19 - CCR7 is found at position 245
#
data_list=dict()
data_list['X']=[]
data_list['Y']=[] 
for i in range (0, 33 ): #len(source_gene_weight) 
    if len(source_gene_weight[i][2]) > 1:
        if "CCR7" in source_gene_weight[i][2] and  ("CCL19" in source_gene_weight[i][0] or "CCL21" in source_gene_weight[i][0]):
            data_list['X'].append(source_gene_weight[i][0] + " - gene_group with " + "CCR7")
        else:
            data_list['X'].append(source_gene_weight[i][0] + " - " + "gene_group")
        
    else:
        data_list['X'].append(source_gene_weight[i][0] + " - " + source_gene_weight[i][2][0])
        
    data_list['Y'].append(source_gene_weight[i][1])
    
data_list_pd = pd.DataFrame({
    'Source Gene': data_list['X'],
    'Gene Importance by NicheCompass': data_list['Y']
})

chart = alt.Chart(data_list_pd).mark_bar().encode(
    x=alt.X("Source Gene:N", axis=alt.Axis(labelAngle=45),sort='-y'),
    y='Gene Importance by NicheCompass'
)

chart.save('/cluster/home/t116508uhn/nichecompass_lymph_source.html')



###########################
#    if "CCR7" in  gp_summary_df_active["gp_target_genes"][index] and ("CCL19" in  gp_summary_df_active["gp_source_genes"][index] or "CCL21" in  gp_summary_df_active["gp_source_genes"][index]): 
#        print(gp_summary_df_active.loc[[index]])

source_target_gene_weight = []
for i in range (0, len(gp_summary_df_active)):
    index = gp_summary_df_active.index[i]
    if len(gp_summary_df_active["gp_source_genes"][index])>1:
        avg_importance_source = np.mean(gp_summary_df_active["gp_source_genes_importances"][index])
        source_gene = gp_summary_df_active["gp_name"][index]
    else:
        avg_importance_source = gp_summary_df_active["gp_source_genes_importances"][index][0]
        source_gene = gp_summary_df_active["gp_source_genes"][index][0]
        
    if len(gp_summary_df_active["gp_target_genes"][index])>1:
        avg_importance_target = np.mean(gp_summary_df_active["gp_target_genes_importances"][index])
    else:
        avg_importance_target = gp_summary_df_active["gp_target_genes_importances"][index][0]

    total_importance = avg_importance_source*avg_importance_target
    
    source_target_gene_weight.append([source_gene, gp_summary_df_active["gp_target_genes"][index], total_importance, index])

    
source_target_gene_weight = sorted(source_target_gene_weight, key = lambda x: x[2], reverse=True)
for i in range (0, len(source_target_gene_weight)):
    if "CCR7" in source_target_gene_weight[i][1] and ("CCL19" in source_target_gene_weight[i][0] or "CCL21" in source_target_gene_weight[i][0]):
        print(i)

#CCL21 - CCR7 is found at position 36
#CCL19 - CCR7 is found at position 234

data_list=dict()
data_list['X']=[]
data_list['Y']=[] 
for i in range (0, 37 ): #len(source_target_gene_weight) 
    if len(source_target_gene_weight[i][1]) > 1:
        if "CCR7" in source_target_gene_weight[i][1] and ("CCL19" in source_target_gene_weight[i][0] or "CCL21" in source_target_gene_weight[i][0]):
            data_list['X'].append(source_target_gene_weight[i][0] + " - gene_group with " + "CCR7")
        else:
            data_list['X'].append(source_target_gene_weight[i][0] + " - " + "gene_group")
    else:
        data_list['X'].append(source_target_gene_weight[i][0] + " - " + source_target_gene_weight[i][1][0])
    
    data_list['Y'].append(source_target_gene_weight[i][2])
    
data_list_pd = pd.DataFrame({
    'Source Gene': data_list['X'],
    'Gene Importance by NicheCompass': data_list['Y']
})

chart = alt.Chart(data_list_pd).mark_bar().encode(
    x=alt.X("Source Gene:N", axis=alt.Axis(labelAngle=45),sort='-y'),
    y='Gene Importance by NicheCompass'
)

chart.save('/cluster/home/t116508uhn/nichecompass_lymph_source_target.html')




# Set parameters for differential gp testing
selected_cats = ["1"]
comparison_cats = "rest"
title = f"NicheCompass Strongly Enriched Niche GPs"
log_bayes_factor_thresh = 2.3
save_fig = True
file_path = f"{figure_folder_path}/" \
            f"/log_bayes_factor_{log_bayes_factor_thresh}" \
             "_niches_enriched_gps_heatmap.svg"

# Run differential gp testing
enriched_gps = model.run_differential_gp_tests(
    cat_key=latent_cluster_key,
    selected_cats=selected_cats,
    comparison_cats=comparison_cats,
    log_bayes_factor_thresh=log_bayes_factor_thresh)

# Results are stored in a df in the adata object
model.adata.uns[differential_gp_test_results_key]


