library(MASS)
library(data.table)
library(cytosignal)
library(Matrix)
library(plot3D)

countsData <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_raw_gene_vs_cell.csv',row.names = 1) 
countsData <- as.matrix(countsData)

## The spatial coordinates need to be presented as a matrix object
spatialData <- as.matrix(read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_spatial.csv', row.names = 1))


csData <- createCytoSignal(raw.data = countsData, cells.loc = spatialData)
csData <- addIntrDB(csData, g_to_u, db.diff, db.cont, inter.index)
csData <- removeLowQuality(csData, counts.thresh = 300)
csData <- changeUniprot(csData)
csData <- inferEpsParams(csData, scale.factor = .73)
csData@parameters$r.diffuse.scale
csData@parameters$sigma.scale
csData <- findNN(csData)
csData <- imputeLR(csData)
csData <- inferIntrScore(csData)

csData <- inferSignif(csData, p.value = 0.05, reads.thresh = 100, sig.thresh = 100)
csData <- rankIntrSpatialVar(csData)
allIntrs <- showIntr(csData, slot.use = "GauEps-Raw", signif.use = "result.spx", return.name = TRUE) 

print(head(allIntrs))
intr.use <- names(allIntrs)[2]

plotEdge(csData, intr.use, slot.use = "GauEps-Raw", pt.size = 0.3, plot.fmt = "svg", return.plot = FALSE, plot_dir = "/", filename = 'cytosignal_human_lymph_ccl19_ccr7.svg')
####################################### our database #####################################################################################


library(MASS)
library(data.table)
library(cytosignal)
library(Matrix)
library(plot3D)

countsData <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_raw_gene_vs_cell.csv',row.names = 1) 
countsData <- as.matrix(countsData)

## The spatial coordinates need to be presented as a matrix object
spatialData <- as.matrix(read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_spatial.csv', row.names = 1))

########################################################################################################################################
g_to_u <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_gene_to_u.csv") #, row.names = 1

inter.index <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_inter.index.csv", row.names = 1)
###################################################################################################
protein_name_ligand <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_protein_name_ligand.csv")
protein_name_ligand <- as.character(protein_name_ligand[,1])
interaction_id_ligand <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_interaction_id_ligand.csv")
interaction_id_ligand <- as.character(interaction_id_ligand[,1])

protein_name_receptor <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_protein_name_receptor.csv")
protein_name_receptor <- as.character(protein_name_receptor[,1])
interaction_id_receptor <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_interaction_id_receptor.csv')
interaction_id_receptor <- as.character(interaction_id_receptor[,1])

# Create factor
ligand_factor <- factor(interaction_id_ligand)
# Set names for the factor
names(ligand_factor) <- protein_name_ligand

# Create factor
receptor_factor <- factor(interaction_id_receptor)
# Set names for the factor
names(receptor_factor) <- protein_name_receptor

combined_factor <- c(ligand_factor, receptor_factor)

db.diff<- list(combined = combined_factor, ligands = ligand_factor, receptors=receptor_factor)

###############################################################################################################################
protein_name_ligand <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_protein_name_ligand.csv")
protein_name_ligand <- as.character(protein_name_ligand[,1])
interaction_id_ligand <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_interaction_id_ligand.csv")
interaction_id_ligand <- as.character(interaction_id_ligand[,1])

protein_name_receptor <- read.csv("/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_protein_name_receptor.csv")
protein_name_receptor <- as.character(protein_name_receptor[,1])
interaction_id_receptor <- read.csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_interaction_id_receptor.csv')
interaction_id_receptor <- as.character(interaction_id_receptor[,1])

# Create factor
ligand_factor <- factor(interaction_id_ligand)
# Set names for the factor
names(ligand_factor) <- protein_name_ligand

# Create factor
receptor_factor <- factor(interaction_id_receptor)
# Set names for the factor
names(receptor_factor) <- protein_name_receptor

combined_factor <- c(ligand_factor, receptor_factor)

db.cont<- list(combined = combined_factor, ligands = ligand_factor, receptors=receptor_factor)
################################################################################################################
csData <- createCytoSignal(raw.data = countsData, cells.loc = spatialData)
csData <- addIntrDB(csData, g_to_u, db.diff, db.cont, inter.index)
csData <- removeLowQuality(csData, counts.thresh = 300)
csData <- changeUniprot(csData)
csData <- inferEpsParams(csData, scale.factor = .73)
csData@parameters$r.diffuse.scale
csData@parameters$sigma.scale
csData <- findNN(csData)
csData <- imputeLR(csData)
csData <- inferIntrScore(csData)

csData <- inferSignif(csData, p.value = 0.05, reads.thresh = 100, sig.thresh = 100)
csData <- rankIntrSpatialVar(csData)
allIntrs <- showIntr(csData, slot.use = "GauEps-Raw", signif.use = "result.spx", return.name = TRUE) 

print(head(allIntrs))
intr.use <- names(allIntrs)[1]

plotEdge(csData, intr.use, slot.use = "GauEps-Raw", pt.size = 0.3, plot.fmt = "svg", return.plot = FALSE, plot_dir = "/cluster/projects/schwartzgroup/fatema/cytosignal/", filename = 'cytosignal_human_lymph_WNT2B-FZD7.svg')





