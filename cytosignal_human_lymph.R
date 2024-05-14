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

cat('write done', '\n')











