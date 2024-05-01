library(MASS)
library(data.table)

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


i_list <- list()
j_list <- list()
score_list <- list()
ccc_list <- list()

for(intr.use in names(allIntrs)){
    cat(intr.use, '\n')
    res.list <- cs@lrscore[["GauEps-Raw"]]@res.list[["result.spx"]]
    lig.slot <- cs@lrscore[["GauEps-Raw"]]@lig.slot
    cells.loc <- as.data.frame(cs@cells.loc) #row=cell id, columns = x,y coordinate
    nn.graph <- cs@imputation[[lig.slot]]@nn.graph #4623 x 4623 sparse Matrix of class "dgCMatrix"
    intrx <- intr.use[1]
    receiver.cells <- res.list[[intrx]] # cell ids
    receiver.idx <- sort(match(receiver.cells, rownames(cells.loc))) # index of those ids in the cell.loc so that you retrieve their x, y
    nn.graph.sig <- nn.graph[, receiver.idx] #4623 x 494 sparse Matrix of class "dgCMatrix"
    # row: from, col:receiver
    # columns/rec are found for a given pair. Now need rows/senders only.
    senders <- unique(nn.graph.sig@i) + 1
    sender_vs_rec <- nn.graph.sig[senders,] # 1901 x 494
    sender_vs_rec <- matrix <- as.matrix(sender_vs_rec) # 0 = no ccc, > 0 = yes CCC
    write.matrix(sender_vs_rec, file="sender_vs_rec.csv")
    mat <- sender_vs_rec
    
    for (i in 1:nrow(mat)){
     
        # looping through columns
        for(j in 1:ncol(mat)){
         
            # check if element is non 
              # zero
            if(mat[i,j]>0.001){
             
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
fwrite(x, file="sender_vs_rec.csv")


#plotEdge(cs, intr.use, slot.use = "GauEps-Raw", pt.size = 0.15, return.plot = FALSE)

res.list <- cs@lrscore[["GauEps-Raw"]]@res.list[["result.spx"]]
lig.slot <- cs@lrscore[["GauEps-Raw"]]@lig.slot
cells.loc <- as.data.frame(cs@cells.loc) #row=cell id, columns = x,y coordinate
nn.graph <- cs@imputation[[lig.slot]]@nn.graph #4623 x 4623 sparse Matrix of class "dgCMatrix"
intrx <- intr.use[1]
receiver.cells <- res.list[[intrx]] # cell ids
receiver.idx <- sort(match(receiver.cells, rownames(cells.loc))) # index of those ids in the cell.loc so that you retrieve their x, y
nn.graph.sig <- nn.graph[, receiver.idx] #4623 x 494 sparse Matrix of class "dgCMatrix"
# row: from, col:receiver
# columns/rec are found for a given pair. Now need rows/senders only.
senders <- unique(nn.graph.sig@i) + 1
sender_vs_rec <- nn.graph.sig[senders,] # 1901 x 494
sender_vs_rec <- matrix <- as.matrix(sender_vs_rec) # 0 = no ccc, > 0 = yes CCC
write.matrix(sender_vs_rec, file="sender_vs_rec.csv")
mat <- sender_vs_rec

i_list <- list()
j_list <- list()
score_list <- list()
ccc_list <- list()

for (i in 1:nrow(mat)){
 
    # looping through columns
    for(j in 1:ncol(mat)){
     
        # check if element is non 
          # zero
        if(mat[i,j]>0.00001){
         
            # display the row and column
              # index
            cat(i, j, mat[i,j], "\n")   
            i_list <- append(i_list, i)
            j_list <- append(j_list, j)
            score_list <- append(score_list, mat[i,j])
            ccc_list <- append(ccc_list, intrx)
            
        }
    }
}
x <- data.table(i=i_list, j=j_list, score=score_list, ccc=ccc_list)
fwrite(x, file="sender_vs_rec.csv")
  
# find sender
senders <- unique(nn.graph.sig@i) + 1 # seems like senders index
sender.idx <- nn.graph.sig@i + 1

seg.up.x <- cells.loc[senders, "x"]
seg.up.y <- cells.loc[senders, "y"]





plotSignif2(cs, intr = intr.use, slot.use = "GauEps-Raw", return.plot = FALSE, edge = TRUE, pt.size = 0.2)



