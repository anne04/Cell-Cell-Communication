## The RDS file will be loaded into a ready-to-use object
dge <- readRDS("SCP2170_annotated_dgCMatrix.rds")

## The cluster annotation need to be presented as a factor object
cluster <- read.csv("SCP2170_cluster.csv")
cluster <- factor(cluster$cell_type)
names(cluster) <- colnames(dge)

## The spatial coordinates need to be presented as a matrix object
spatial <- as.matrix(read.csv("SCP2170_spatial.csv", row.names = 1))
## Please make sure that the dimension names are lower case "x" and "y"
colnames(spatial) <- c("x", "y")

library(cytosignal)

cs <- createCytoSignal(raw.data = dge, cells.loc = spatial, clusters = cluster)
cs <- addIntrDB(cs, g_to_u, db.diff, db.cont, inter.index)
cs <- removeLowQuality(cs, counts.thresh = 300)
cs <- changeUniprot(cs)
cs <- inferEpsParams(cs, scale.factor = 0.73)
cs@parameters$r.diffuse.scale
cs@parameters$sigma.scale
cs <- findNN(cs)
cs <- imputeLR(cs)
cs <- inferIntrScore(cs)
cs <- inferSignif(cs, p.value = 0.05, reads.thresh = 100, sig.thresh = 100)
cs <- rankIntrSpatialVar(cs)

allIntrs <- showIntr(cs, slot.use = "GauEps-Raw", signif.use = "result.spx", return.name = TRUE)
print(head(allIntrs))
intr.use <- names(allIntrs)[1]

plotEdge(cs, intr.use, slot.use = "GauEps-Raw", pt.size = 0.15, return.plot = FALSE)

plotSignif2(cs, intr = intr.use, slot.use = "GauEps-Raw", return.plot = FALSE, edge = TRUE, pt.size = 0.2)



