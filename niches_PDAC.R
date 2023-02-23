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
                           CellToCellSpatial = F,CellToNeighborhood = F,NeighborhoodToCell = T)
                           
                           
niche <- NICHES_output[['NeighborhoodToCell']]
Idents(niche) <- niche[['ReceivingType']]

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

  
