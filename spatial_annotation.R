library(reticulate)
np <- import("numpy")
# data reading
mat <- np$load("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_quantile_transformed.npy")

