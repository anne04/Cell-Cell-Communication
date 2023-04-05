library(reticulate)
np <- import("numpy")
# data reading
mat <- np$load("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_quantile_transformed.npy")
# nrow(mat) is 19523 and ncol(mat) is 1406.


cell_barcodes <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cell_barcode.csv', header = FALSE)
# nrow(cell_barcodes) is 1406 and ncol(cell_barcodes) is 1.

gene_ids <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/gene_ids.csv', header = FALSE)
# nrow(gene_ids) is 19523 and ncol(gene_ids) is 1.

rownames(mat) <- gene_ids
# Assign cell_barcodes as column names and gene_ids as row names to the mat.   
