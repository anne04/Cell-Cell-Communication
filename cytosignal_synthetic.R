library(MASS)
library(data.table)
library(cytosignal)


options = 'dt-path_equally_spaced_lrc1467_cp100_noise0_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp'
## The RDS file will be loaded into a ready-to-use object
#dge_raw <- readRDS(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_raw_gene_vs_cell_',options,'.csv', sep="")) # rows are genes, columns are cells. gene x cell sparse matrix
countsData <- read.csv(file = paste('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_raw_gene_vs_cell_',options,'.csv', sep=""),row.names = 1) 
countsData <- as.matrix(countsData)

## The cluster annotation need to be presented as a factor object
#cluster <- read.csv("SCP2170_cluster.csv")
#cluster <- factor(cluster$cell_type)
#names(cluster) <- colnames(dge)

## The spatial coordinates need to be presented as a matrix object
spatialData <- as.matrix(read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_cell_',options,'_spatial.csv',sep=""), row.names = 1))
## Please make sure that the dimension names are lower case "x" and "y"
#colnames(spatial) <- c("x", "y")

csData <- createCytoSignal(raw.data = countsData, cells.loc = spatialData)

