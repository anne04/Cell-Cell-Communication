library(Seurat)             
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(SeuratWrappers)
library(NICHES)
library(viridis)

data_dir <- '/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
list.files(data_dir)
seurat_object <- Load10X_Spatial(data.dir = data_dir)
pancreas <- SCTransform(seurat_object, assay = "Spatial", verbose = FALSE)
pancreas <- RunPCA(pancreas, assay = "SCT", verbose = FALSE)
pancreas <- FindNeighbors(pancreas, reduction = "pca", dims = 1:30)
pancreas <- FindClusters(pancreas, verbose = FALSE)
pancreas <- RunUMAP(pancreas, reduction = "pca", dims = 1:30)
p1 <- DimPlot(pancreas, reduction = "umap",group.by = 'seurat_clusters', label = TRUE)
p2 <- SpatialDimPlot(pancreas, label = TRUE,group.by = 'seurat_clusters', label.size = 3)
ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = (p1+p2))

pancreas@meta.data$x <- pancreas@images$slice1@coordinates$row
pancreas@meta.data$y <- pancreas@images$slice1@coordinates$col

DefaultAssay(pancreas) <- "Spatial"
pancreas <- NormalizeData(pancreas)

pancreas <- SeuratWrappers::RunALRA(pancreas)
lr_db <- read.csv("/cluster/home/t116508uhn/64630/lr_cellchat_nichenet.csv")
NICHES_output <- RunNICHES(object = pancreas,
                           LR.database = "custom",
                           custom_LR_database = lr_db,
                           species = "human",
                           assay = "alra",
                           position.x = 'x',
                           position.y = 'y',
                           k = 12, 
                           cell_types = "seurat_clusters",
                           min.cells.per.ident = 0,
                           min.cells.per.gene = NULL,
                           meta.data.to.map = c('orig.ident','seurat_clusters'),
                           CellToCell = F,CellToSystem = F,SystemToCell = F,
                           CellToCellSpatial = T,CellToNeighborhood = F,NeighborhoodToCell = F)
                           
                           
niche <- NICHES_output[['CellToCellSpatial']]
Idents(niche) <- niche[['ReceivingType']]


cc.object <- NICHES_output$CellToCellSpatial #Extract the output of interest
cc.object <- ScaleData(cc.object) #Scale
#cc.object <- FindVariableFeatures(cc.object,selection.method="disp") #Identify variable features
#cc.object <- RunPCA(cc.object,npcs = 100) #RunPCA
#cc.object <- RunUMAP(cc.object,dims = 1:100)
Idents(cc.object) <- cc.object[['ReceivingType']]
ec.network <- subset(cc.object,idents ='8')
Idents(ec.network) <- ec.network[['VectorType']]
mark.ec <- FindAllMarkers(ec.network,
                          logfc.threshold = 1,
                          min.pct = 0.5,
                          only.pos = T,
                          test.use = 'roc')
# Pull markers of interest to plot
mark.ec$ratio <- mark.ec$pct.1/mark.ec$pct.2
marker.list.ec <- mark.ec %>% group_by(cluster) %>% top_n(5,avg_log2FC)
#p <- DoHeatmap(ec.network,features = marker.list.ec$gene,cells = WhichCells(ec.network,downsample = 100))
#ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)

temp_matrix = marker.list.ec[['gene']] #mark.ec[['gene']]
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_pairs_8_brief.csv')


temp_matrix = mark.ec[['gene']]
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_pairs_8.csv')



temp_matrix = GetAssayData(object = niche, slot = "counts")
temp_matrix = as.matrix(temp_matrix)
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_PDAC_pair_vs_cells.csv')

temp_matrix = niche[['seurat_clusters.Joint_clusters']]
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_PDAC_cluster_vs_cells.csv')



# Scale and visualize
niche <- ScaleData(niche)
niche <- FindVariableFeatures(niche,selection.method = "disp")
niche <- RunPCA(niche)
p <- ElbowPlot(niche,ndims = 50)
ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)


niche <- RunUMAP(niche,dims = 1:10)  
p <- DimPlot(niche,reduction = 'umap',pt.size = 0.5,shuffle = T, label = T) +ggtitle('Cellular Microenvironment')+NoLegend()
ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)


mark <- FindAllMarkers(niche,min.pct = 0.25,only.pos = T,test.use = "roc")
GOI_niche <- mark %>% group_by(cluster) %>% top_n(5,myAUC)
p <- DoHeatmap(niche,features = unique(GOI_niche$gene))+ scale_fill_gradientn(colors = c("grey","white", "blue")) 
ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)

##########################################################################################
df=read.csv(file = '/cluster/home/t116508uhn/synthetic_cell_type6_f_x.csv', header = FALSE)
cell_x=list()  
for(i in 1:ncol(df)) {      
  cell_x[[i]] <- df[ , i]    
}
df=read.csv(file = '/cluster/home/t116508uhn/synthetic_cell_type6_f_y.csv', header = FALSE)
cell_y=list()  
for(i in 1:ncol(df)) {      
  cell_y[[i]] <- df[ , i]    
}

countsData <- read.csv(file = '/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_f.csv',row.names = 1)
pdac_sample <- CreateSeuratObject(counts = countsData)
temp <- SCTransform(pdac_sample, verbose = FALSE)
#DefaultAssay(temp) <- "integrated"
temp <- RunPCA(temp, verbose = FALSE)
temp <- FindNeighbors(temp, reduction = "pca", dims = 1:30)
temp <- FindClusters(temp, verbose = FALSE)
temp <- RunUMAP(temp , reduction = "pca", dims = 1:30)

#temp@images$slice1@coordinates$row <- cell_x[[1]]
#temp@images$slice1@coordinates$col <- cell_y[[1]]

#p1 <- DimPlot(temp , reduction = "umap",group.by = 'seurat_clusters', label = TRUE)
#p2 <- SpatialDimPlot(temp , label = TRUE,group.by = 'seurat_clusters', label.size = 3)
#ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = (p1+p2))
temp@meta.data$x <- cell_x[[1]]
temp@meta.data$y <- cell_y[[1]]
#DefaultAssay(temp) <- "Spatial"
temp <- NormalizeData(temp)

temp <- SeuratWrappers::RunALRA(temp)

lr_db <- read.csv("/cluster/home/t116508uhn/synthetic_lr_type6_f.csv")
NICHES_output <- RunNICHES(object = temp,
                           LR.database = "custom",
                           custom_LR_database = lr_db,
                           species = "human",
                           assay = "alra",
                           position.x = 'x',
                           position.y = 'y',
                           k = 20, 
                           cell_types = "seurat_clusters",
                           min.cells.per.ident = 0,
                           min.cells.per.gene = NULL,
                           meta.data.to.map = c('orig.ident','seurat_clusters'),
                           CellToCell = F,CellToSystem = F,SystemToCell = F,
                           CellToCellSpatial = T, CellToNeighborhood = F,NeighborhoodToCell = F)
        
niche <- NICHES_output[['CellToCellSpatial']]
Idents(niche) <- niche[['ReceivingType']]


#temp_matrix = GetAssayData(object = niche, slot = "counts")
#temp_matrix = as.matrix(temp_matrix)
#write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_pair_vs_cells_type6_e.csv')



# Scale and visualize
niche <- ScaleData(niche)
niche <- FindVariableFeatures(niche,selection.method = "disp")
niche <- RunPCA(niche)
#p <- ElbowPlot(niche,ndims = 50)
#ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)

niche <- RunUMAP(niche,dims = 1:6)  
#p <- DimPlot(niche,reduction = 'umap',pt.size = 0.5,shuffle = T, label = T) +ggtitle('Cellular Microenvironment')+NoLegend()
#ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)
#mark <- FindAllMarkers(niche,min.pct = 0.25,only.pos = T,test.use = "roc")
#GOI_niche <- mark %>% group_by(cluster) %>% top_n(5,myAUC)
#p <- DoHeatmap(niche,features = unique(GOI_niche$gene))+ scale_fill_gradientn(colors = c("grey","white", "blue")) 
#ggsave("/cluster/home/t116508uhn/64630/myplot.png", plot = p)

temp_matrix = GetAssayData(object = niche, slot = "counts")
temp_matrix = as.matrix(temp_matrix)
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_pair_vs_cells_type6_f.csv')

temp_matrix = niche[['seurat_clusters.Joint_clusters']]
write.csv(temp_matrix, '/cluster/home/t116508uhn/niches_output_cluster_vs_cells.csv')

