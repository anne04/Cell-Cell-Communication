library(reticulate)
library(SingleR)

#' Subtype single cells through SingleR, using bulk tumours as reference.
#' 
#' @param mat Count matrix where genes (HGNC format) are rows and cells are columns. LogCPM values are recommended. Can be a sparse matrix.
#' @param reference_file Path to reference bulk tumour RData containing the objects ref_mat (LogTPM HGNC gene by sample matrix) and ref_labels (named character vector, sample: subtype label). Default "bulk_reference.RData".
#' @param find_mixed Identify cells of the Mixed subtype (formerly also called Hybrid subtype). Default TRUE.
#' @param ... Additional arguments passed to SingleR
#' 
#' @return List with the following: 
#' 1) data: data.frame where rows are cells and columns represent the subtype, normalized confidence score and normalized subtype scores
#' 2) singler_pred: SingleR raw output
#' 
singler_subtype <- function(mat, reference_file="/cluster/projects/schwartzgroup/fatema/sabrina/bulk_reference.RData", find_mixed=TRUE, ...) {

  load(reference_file)
  if (!all(colnames(ref_mat) == names(ref_labels))) {
    stop("Mismatched sample names in reference file. Please check if you are using the correct reference.")
  }
  if ( !("ClassicA" %in% ref_labels) | !("BasalB" %in% ref_labels) | "hybrid" %in% ref_labels) {
    stop("Reference subtype labels do not match expected. Please check if you are using the correct reference.")
  }
  
  overlap <- length(intersect(rownames(mat), rownames(ref_mat))) / nrow(ref_mat)
  if (overlap < 0.2) {
    message("Reference and query gene overlap is low (", round(overlap * 100, 2), "%). Please check if query single cell matrix contains genes (as rownames) in HGNC format.")
  }
  
  pred <- SingleR(test=mat, ref=ref_mat, labels=ref_labels, ...)
  
  # Normalize SingleR scores
  score_metadata <- as.data.frame(pred$scores)
  colnames(score_metadata) <- paste0(colnames(score_metadata), "_score")
  rownames(score_metadata) <- rownames(pred)
  score_conf <- rowMeans(score_metadata)
  score_metadata_norm <- score_metadata - score_conf
  
  data <- cbind(subtype=pred$labels, 
                conf=score_conf,
                score_metadata_norm)
  
  if (find_mixed) {
    mixed_cutoff <- 0.02
    # Cells identified as Mixed if originally ClassicA or BasalB, and similar in score for both.
    subtype_with_mixed <- ifelse(data$subtype %in% c("ClassicA", "BasalB") & 
                                   abs(data$ClassicA_score - data$BasalB_score) < mixed_cutoff, 
                                 "Mixed", data$subtype) 
    data$subtype <- subtype_with_mixed
  }

  return(list(data=data, singler_pred=pred))
  
}



##############################################################################
np <- import("numpy")
# data reading
mat <- np$load("/cluster/projects/schwartzgroup/fatema/find_ccc/gene_vs_cell_quantile_transformed.npy")
# nrow(mat) is 19523 and ncol(mat) is 1406.


cell_barcodes <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cell_barcode.csv', header = FALSE)
# nrow(cell_barcodes) is 1406 and ncol(cell_barcodes) is 1.

gene_ids <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/gene_ids.csv', header = FALSE)
# nrow(gene_ids) is 19523 and ncol(gene_ids) is 1.

cell_vector <- cell_barcodes[, 1]
gene_vector <- gene_ids[, 1]

colnames(mat) <- cell_vector
rownames(mat) <- gene_vector
# Assign cell_barcodes as column names and gene_ids as row names to the mat.   

spot_prediction = singler_subtype(mat)

write.csv(spot_prediction, '/cluster/projects/schwartzgroup/fatema/find_ccc/singleR_spot_annotation_Sabrina.csv')

