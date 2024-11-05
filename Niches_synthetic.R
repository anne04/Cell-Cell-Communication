# https://msraredon.github.io/NICHES/articles/03%20Rat%20Alveolus.html
# https://msraredon.github.io/NICHES/articles/01%20NICHES%20Spatial.html
library(Seurat)             
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(SeuratWrappers)
library(NICHES)
library(viridis)


####################################### synthetic ###################################################
options = 'mix'
savepath = '/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/'
df=read.csv(file = paste("/cluster/home/t116508uhn/synthetic_cell_",options,"_x.csv",sep=""), header = FALSE) #read.csv(file = '/cluster/home/t116508uhn/synthetic_cell_type6_f_x.csv', header = FALSE)
cell_x=list()  
for(i in 1:ncol(df)) {      
  cell_x[[i]] <- df[ , i]    
}
df=read.csv(file = paste('/cluster/home/t116508uhn/synthetic_cell_',options,'_y.csv',sep=""), header = FALSE) #read.csv(file = '/cluster/home/t116508uhn/synthetic_cell_type6_f_y.csv', header = FALSE)
cell_y=list()  
for(i in 1:ncol(df)) {      
  cell_y[[i]] <- df[ , i]    
}

countsData <- read.csv(file = paste('/cluster/home/t116508uhn/synthetic_gene_vs_cell_',options,'.csv', sep=""),row.names = 1) # read.csv(file = '/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_f.csv',row.names = 1)
pdac_sample <- CreateSeuratObject(counts = countsData)
#temp <- SCTransform(pdac_sample)
temp <- ScaleData(pdac_sample)
temp <- FindVariableFeatures(temp) 
temp <- RunPCA(temp, verbose = FALSE)
temp <- FindNeighbors(temp, reduction = "pca", dims = 1:30)
temp <- FindClusters(temp, verbose = FALSE)
temp <- RunUMAP(temp , reduction = "pca", dims = 1:30)

temp@meta.data$x <- cell_x[[1]]
temp@meta.data$y <- cell_y[[1]]
#DefaultAssay(temp) <- "Spatial"

temp <- NormalizeData(temp)

temp <- SeuratWrappers::RunALRA(temp)

lr_db <- read.csv(paste("/cluster/home/t116508uhn/synthetic_lr_",options,".csv",sep=""))
NICHES_output <- RunNICHES(object = temp,
                           LR.database = "custom",
                           custom_LR_database = lr_db,
                           species = "human",
                           assay = "alra",
                           position.x = 'x',
                           position.y = 'y',
                           k = 30, 
                           cell_types = "seurat_clusters",
                           min.cells.per.ident = 0,
                           min.cells.per.gene = NULL,
                           meta.data.to.map = c('orig.ident','seurat_clusters'),
                           CellToCell = F,CellToSystem = F,SystemToCell = F,
                           CellToCellSpatial = T, CellToNeighborhood = F,NeighborhoodToCell = F)
        
niche <- NICHES_output[['CellToCellSpatial']]
niche <- ScaleData(niche)
niche <- FindVariableFeatures(niche,selection.method = "disp")
niche <- RunPCA(niche)
niche <- RunUMAP(niche,dims = 1:10)   # same as number of pca

#### save scaled coexpression score matrix 
temp_matrix = GetAssayData(object = niche, slot = "scale.data") #https://satijalab.org/seurat/articles/essential_commands.html#data-access
temp_matrix = as.matrix(temp_matrix)
write.csv(temp_matrix, paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_output_pair_vs_cells_',options,'.csv',sep=""))

############################## print marker genes #######################################
Idents(niche) <- niche[['ReceivingType']]
ec.network <- niche
Idents(ec.network) <- ec.network[['VectorType']]
mark.ec <- FindAllMarkers(ec.network,
                          logfc.threshold = 1,
                          min.pct = 0.5,
                          only.pos = T,
                          test.use = 'roc')

# Pull markers of interest to plot
mark.ec$ratio <- mark.ec$pct.1/mark.ec$pct.2

marker.list.ec <- mark.ec %>% group_by(cluster) %>% top_n(5,avg_log2FC) #
write.csv(marker.list.ec, paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_output_ccc_lr_pairs_markerList_top5_',options,'.csv',sep=""))

write.csv(ec.network[['VectorType']], paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_VectorType_',options,'.csv',sep=""))

#features = unique(marker.list.ec$gene)
#write.csv(features, paste('/cluster/home/t116508uhn/niches_output_ccc_lr_pairs_top5_',options,'.csv',sep=""))

#cells = WhichCells(ec.network)
#write.csv(cells, paste('/cluster/home/t116508uhn/niches_output_ccc_cells_',options,'.csv',sep=""))

#cells = WhichCells(ec.network,downsample = 100)
#write.csv(cells, paste('/cluster/home/t116508uhn/niches_output_ccc_cells_downsampled_',options,'.csv',sep=""))

################################# if above does not work then do this ###############################
Idents(niche) <- niche[['ReceivingType']]
ec.network <- niche
Idents(ec.network) <- ec.network[['VectorType']]
mark.ec <- FindAllMarkers(ec.network,min.pct = 0.25,only.pos = T,test.use = "roc") 

marker.list.ec <- mark.ec %>% group_by(cluster) %>% top_n(5,myAUC) #
write.csv(marker.list.ec, paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_output_ccc_lr_pairs_markerList_top5_',options,'.csv',sep=""))
# write.csv(mark.ec, paste('/cluster/home/t116508uhn/niches_output_ccc_lr_pairs_markerList_top5_',options,'.csv',sep=""))

write.csv(ec.network[['VectorType']], paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_VectorType_',options,'.csv',sep=""))

#features = unique(marker.list.ec$gene)
#write.csv(features, paste('/cluster/home/t116508uhn/niches_output_ccc_lr_pairs_top5_',options,'.csv',sep=""))

#cells = WhichCells(ec.network)
#write.csv(cells, paste('/cluster/home/t116508uhn/niches_output_ccc_cells_',options,'.csv',sep=""))

#cells = WhichCells(ec.network,downsample = 100)
#write.csv(cells, paste('/cluster/home/t116508uhn/niches_output_ccc_cells_downsampled_',options,'.csv',sep=""))


################################################################################################################

#p <- DoHeatmap(ec.network,features = marker.list.ec$gene,cells = WhichCells(ec.network,downsample = 100))


#temp_matrix = GetAssayData(object = niche, slot = "counts")
#temp_matrix = as.matrix(temp_matrix)
#write.csv(temp_matrix, paste('/cluster/home/t116508uhn/niches_output_pair_vs_cells_',options,'.csv',sep=""))


mark <- FindAllMarkers(niche,min.pct = 0.25,only.pos = T,test.use = "roc")
GOI_niche <- mark %>% group_by(cluster) %>% top_n(5,myAUC)

temp_matrix = niche[['seurat_clusters.Joint_clusters']]
write.csv(temp_matrix, paste('/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/niches_output_cluster_vs_cells_',options,'.csv',sep=""))
