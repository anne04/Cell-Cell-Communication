library(MASS)
library(data.table)
library(cytosignal)
library(Matrix)

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


g_to_u <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/gene_to_u_',options,'.csv',sep="")) #, row.names = 1

inter.index <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/inter.index_',options,'.csv',sep=""), row.names = 1)
###############################################################################################################################
protein_name_ligand <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/protein_name_ligand_',options,'.csv',sep=""))
protein_name_ligand <- as.character(protein_name_ligand[,1])
interaction_id_ligand <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/interaction_id_ligand_',options,'.csv',sep=""))
interaction_id_ligand <- as.character(interaction_id_ligand[,1])

protein_name_receptor <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/protein_name_receptor_',options,'.csv',sep=""))
protein_name_receptor <- as.character(protein_name_receptor[,1])
interaction_id_receptor <- read.csv(paste('/cluster/home/t116508uhn/cytosignal/interaction_id_receptor_',options,'.csv',sep=""))
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

####################################################################
###############################################################################################################################
protein_name_ligand <- as.character()
interaction_id_ligand <- as.character()
protein_name_receptor <- as.character()
interaction_id_receptor <- as.character()

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
###########################################################################################################################

csData <- createCytoSignal(raw.data = countsData, cells.loc = spatialData)
db.cont <- db.diff
csData <- addIntrDB(csData, g_to_u, db.diff, db.cont, inter.index)
csData <- removeLowQuality(csData, counts.thresh = 1, gene.thresh = 1)
csData <- changeUniprot(csData)
csData <- inferEpsParams(csData, scale.factor = 100)
csData@parameters$r.diffuse.scale
csData@parameters$sigma.scale
csData <- findNN(csData)
csData <- imputeLR(csData)
csData <- inferIntrScore(csData)

csData <- inferSignif(csData, p.value = 0.05, reads.thresh = 100, sig.thresh = 100)
csData <- rankIntrSpatialVar(csData)

allIntrs <- showIntr(csData, slot.use = "GauEps-Raw", signif.use = "result.spx") #, return.name = TRUE
print(head(allIntrs))
intr.use <- names(allIntrs)[1]

i_list <- list()
j_list <- list()
score_list <- list()
ccc_list <- list()

for(intr.use in allIntrs){ #names(allIntrs)
    cat(intr.use, '\n')
    res.list <- csData@lrscore[["GauEps-Raw"]]@res.list[["result.spx"]]
    lig.slot <- csData@lrscore[["GauEps-Raw"]]@lig.slot
    cells.loc <- as.data.frame(csData@cells.loc) #row=cell id, columns = x,y coordinate
    nn.graph <- csData@imputation[[lig.slot]]@nn.graph #4623 x 4623 sparse Matrix of class "dgCMatrix"
    intrx <- intr.use #[1]
    receiver.cells <- res.list[[intrx]] # cell ids
    receiver.idx <- sort(match(receiver.cells, rownames(cells.loc))) # index of those ids in the cell.loc so that you retrieve their x, y
    nn.graph.sig <- nn.graph[, receiver.idx] #4623 x 494 sparse Matrix of class "dgCMatrix"
    # row: from, col:receiver
    # columns/rec are found for a given pair. Now need rows/senders only.
    senders <- unique(nn.graph.sig@i) + 1
    sender_vs_rec <- nn.graph.sig[senders,] # 1901 x 494
    sender_vs_rec <- as.matrix(sender_vs_rec) # 0 = no ccc, > 0 = yes CCC
    #write.matrix(sender_vs_rec, file="sender_vs_rec.csv")
    mat <- sender_vs_rec
    
    for (i in 1:nrow(mat)){
     
        # looping through columns
        for(j in 1:ncol(mat)){
         
            # check if element is non 
              # zero
            if(mat[i,j]>0){
             
                # display the row and column
                  # index
                #cat(i, j, mat[i,j], "\n")   
                i_list <- append(i_list, i)
                j_list <- append(j_list, j)
                score_list <- append(score_list, mat[i,j])
                ccc_list <- append(ccc_list, intrx)
                
            }
        }
    }
    
    
    }
x <- data.table(i=i_list, j=j_list, score=score_list, ccc=ccc_list)
fwrite(x, file=paste('sender_vs_rec_',options,'.csv',sep=""))













