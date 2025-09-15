import spatialdata as sd
from spatialdata_io import xenium

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq

xenium_path = 'data/' #"./Xenium"
zarr_path = "Xenium.zarr"
sdata = xenium(xenium_path)
sdata.write(zarr_path)

sdata = sd.read_zarr(zarr_path)
sdata

adata = sdata.tables["table"]
adata.write('Xenium_Prime_Human_Skin_FFPE.h5ad', compression="gzip")

# module load python/3.11
# python
# then run these