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

g_to_u <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/gene_to_u_',options,'.csv',sep="")) #, row.names = 1


protein_name_ligand <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/protein_name_ligand_',options,'.csv',sep=""))
protein_name_ligand <- as.character(protein_name_ligand[,1])

interaction_id_ligand <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/interaction_id_ligand_',options,'.csv',sep=""))
interaction_id_ligand <- as.character(interaction_id_ligand[,1])

# Create factor
ligand_factor <- factor(interaction_id_ligand)
# Set names for the factor
names(ligand_factor) <- protein_name_ligand

protein_name_receptor <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/protein_name_receptor_',options,'.csv',sep=""))
protein_name_receptor <- as.character(protein_name_receptor[,1])

interaction_id_receptor <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/interaction_id_receptor_',options,'.csv',sep=""))
interaction_id_receptor <- as.character(interaction_id_receptor[,1])

# Create factor
receptor_factor <- factor(interaction_id_receptor)
# Set names for the factor
names(receptor_factor) <- protein_name_receptor

combined_factor <- c(ligand_factor, receptor_factor)

diff.cont <- list(combined = combined_factor, ligands = ligand_factor, receptors=receptor_factor)

