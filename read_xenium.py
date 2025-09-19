import spatialdata as sd
from spatialdata_io import xenium


xenium_path = 'data/outs/' #"./Xenium"
zarr_path = "Xenium.zarr"
sdata = xenium(xenium_path)
sdata.write(zarr_path)

sdata = sd.read_zarr(zarr_path)
sdata

adata = sdata.tables["table"]
#adata.write('Xenium_Prime_Human_Skin_FFPE.h5ad', compression="gzip")
adata.write('Xenium_FFPE_Human_Breast_Cancer_Rep1.h5ad', compression="gzip")
# module load gcc arrow
# module load python/3.11
# python
# then run these