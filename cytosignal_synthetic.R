library(MASS)
library(data.table)

options 
## The RDS file will be loaded into a ready-to-use object
dge <- readRDS("SCP2170_annotated_dgCMatrix.rds") # rows are genes, columns are cells. gene x cell sparse matrix

## The cluster annotation need to be presented as a factor object
cluster <- read.csv("SCP2170_cluster.csv")
cluster <- factor(cluster$cell_type)
names(cluster) <- colnames(dge)

## The spatial coordinates need to be presented as a matrix object
spatial <- as.matrix(read.csv("SCP2170_spatial.csv", row.names = 1))
## Please make sure that the dimension names are lower case "x" and "y"
colnames(spatial) <- c("x", "y")

library(cytosignal)
