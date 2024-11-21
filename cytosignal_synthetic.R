library(MASS)
library(data.table)
library(cytosignal)
library(Matrix)

#options = 'dt-path_equally_spaced_lrc1467_cp100_noise0_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp'
#options =  'dt-path_equally_spaced_lrc1467_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp'
#options = 'dt-path_equally_spaced_lrc1467_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp'

#options = 'dt-path_uniform_distribution_lrc112_cp100_noise0_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp'
#options = 'dt-path_uniform_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp'
#options = 'dt-path_uniform_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp_v2'

#options = 'dt-path_mixture_of_distribution_lrc112_cp100_noise0_random_overlap_knn_cellCount5000_3dim_3patterns_temp'
#options = 'dt-path_mixture_of_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp'
#options = 'dt-path_mixture_of_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp'
# -- done #
#options = 'dt-randomCCC_equally_spaced_lrc105_cp100_noise0_threshold_dist_cellCount3000'
#options = 'dt-randomCCC_uniform_distribution_lrc105_cp100_noise0_threshold_dist_cellCount5000'
#options = 'dt-randomCCC_mix_distribution_lrc105_cp100_noise0_knn_cellCount5000'

#options = 'equidistant_mechanistic_noise0'
#options = 'uniform_mechanistic_noise0'
options = 'mixture_mechanistic_noise0'


countsData <- read.csv(file = paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/synthetic_raw_gene_vs_cell_',options,'.csv', sep=""),row.names = 1) 
countsData <- as.matrix(countsData)

## The cluster annotation need to be presented as a factor object
#cluster <- read.csv("SCP2170_cluster.csv")
#cluster <- factor(cluster$cell_type)
#names(cluster) <- colnames(dge)

## The spatial coordinates need to be presented as a matrix object
spatialData <- as.matrix(read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/synthetic_cell_',options,'_spatial.csv',sep=""), row.names = 1))


g_to_u <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/gene_to_u_',options,'.csv',sep="")) #, row.names = 1

inter.index <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/inter.index_',options,'.csv',sep=""), row.names = 1)
###############################################################################################################################
protein_name_ligand <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/protein_name_ligand_',options,'.csv',sep=""))
protein_name_ligand <- as.character(protein_name_ligand[,1])
interaction_id_ligand <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/interaction_id_ligand_',options,'.csv',sep=""))
interaction_id_ligand <- as.character(interaction_id_ligand[,1])

protein_name_receptor <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/protein_name_receptor_',options,'.csv',sep=""))
protein_name_receptor <- as.character(protein_name_receptor[,1])
interaction_id_receptor <- read.csv(paste('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/interaction_id_receptor_',options,'.csv',sep=""))
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
#  scale.factor = 10 -- controls the TP finding, [pvalue, reads.thresh = 50, sig.thresh = 50 ]-- spatial variablitity
csData <- createCytoSignal(raw.data = countsData, cells.loc = spatialData)
db.cont <- db.diff
csData <- addIntrDB(csData, g_to_u, db.diff, db.cont, inter.index)
csData <- removeLowQuality(csData, counts.thresh = 1, gene.thresh = 1)
csData <- changeUniprot(csData)
csData <- inferEpsParams(csData, scale.factor = 5) 
# 10 for op: 3, 4, 5, which caused diffuse dist = 20, whereas we had just 4. 
# 5 for op = 6, 7, 8 which caused diffusion 40, whereas we had 10nn. 
# 20 for op 0, 1, 2 which caused diffusion = 10, whereas we had 1.2. 
# 5 for op 9, 10, 11 - 40 diffuse
csData@parameters$r.diffuse.scale
csData@parameters$sigma.scale
csData <- findNN(csData)
csData <- imputeLR(csData)
csData <- inferIntrScore(csData)

csData <- inferSignif(csData, p.value = 0.10, reads.thresh = 100, sig.thresh = 100) #0.50
csData <- rankIntrSpatialVar(csData)
allIntrs <- showIntr(csData, slot.use = "GauEps-Raw", signif.use = "result", return.name = TRUE) #

#print(head(allIntrs))
#intr.use <- allIntrs[1] #names(allIntrs)[1]


res.list <- csData@lrscore[["GauEps-Raw"]]@res.list[["result"]] #.spx
lig.slot <- csData@lrscore[["GauEps-Raw"]]@lig.slot # what type of ligands? Diffuse or contact. It is a string type. 
cells.loc <- as.data.frame(csData@cells.loc) #row=cell id, columns = x,y coordinate

nn.graph <- csData@imputation[[lig.slot]]@nn.graph #4623 x 4623 sparse Matrix of class "dgCMatrix" -- cell vs cell -- for the provided ligand type (lig.slot) -- here we use diffuse


write.csv(as.matrix(nn.graph), paste('/cluster/projects/schwartzgroup/fatema/cytosignal/cell_cell_score_',options,'.csv',sep=""))
write.csv(names(allIntrs), paste('/cluster/projects/schwartzgroup/fatema/cytosignal/ccc_name_',options,'.csv',sep=""))

### no need of below ###
i_list <- list()
j_list <- list()
score_list <- list()
ccc_list <- list()
# if there are multiple lr pairs between two cells: from cell a to cell b --> they are combined into one "imputation" score and ONE score/value is assigned to the slot [a][b] --> a to b  
for(intr.use in names(allIntrs)){ 
    cat(intr.use, '\n')
    intrx <- intr.use[1]
    receiver.cells <- res.list[[intrx]] # cell ids
    receiver.idx <- sort(match(receiver.cells, rownames(cells.loc))) # index of those ids in the cell.loc so that you retrieve their x, y
    nn.graph.sig <- nn.graph[, receiver.idx] #4623 x 494 sparse Matrix of class "dgCMatrix" # seperate the columns having those receiver.cells 
    # row: from, col:receiver
    # columns/rec are found for a given pair. Now need rows/senders only.
    senders <- unique(nn.graph.sig@i) + 1 #
    sender_vs_rec <- nn.graph.sig[senders,] # 1901 x 494
    sender_vs_rec <- as.matrix(sender_vs_rec) # 0 = no ccc, > 0 = yes CCC
    #write.matrix(sender_vs_rec, file="sender_vs_rec.csv")
    mat <- sender_vs_rec
    if (nrow(mat)>0)
    {
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
    
    }
x <- data.table(i=i_list, j=j_list, score=score_list, ccc=ccc_list)
fwrite(x, file=paste('sender_vs_rec_',options,'.csv',sep=""))

cat('write done', '\n')











