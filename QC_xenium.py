
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

xenium_file_path = 'data/Xenium_Prime_Human_Skin_FFPE.h5ad'
adata= sc.read_h5ad(xenium_file_path)
print('input data read done')


# QC plots
plt.clf()
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata.obs["transcript_counts"],
    kde=False,
    ax=axs[1],
)

axs[1].set_xlim(0, 800)

axs[2].set_title("Area of segmented cells")
sns.histplot(
    adata.obs["cell_area"],
    kde=False,
    ax=axs[2],
)

axs[3].set_title("Nucleus ratio")
sns.histplot(
    adata.obs["nucleus_area"] / adata.obs["cell_area"],
    kde=False,
    ax=axs[3],
)

# Save the figure
plt.tight_layout()  # Adjust layout to avoid overlap
fig.savefig("/cluster/home/t116508uhn/LRbind_output/Xenium_Prime_Human_Skin_FFPE_qc_plots.png", dpi=300)  # Save as PNG with high resolution


# QC filtering
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=5)

'''

# Perform PCA, neighbors computation, and Leiden clustering
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.tsne(adata)
# save the data
adata.write("/cluster/projects/schwartzgroup/vg/20241212__190528__Lok_Jalal_241210/23169_processed.h5ad")

'''