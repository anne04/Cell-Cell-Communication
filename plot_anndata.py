import matplotlib.pyplot as plt
import anndata
from collections import defaultdict
import altair as alt
import pandas as pd

output_name = "data/Xenium_Prime_Human_Skin_FFPE"
sample_name = 'data/Xenium_Prime_Human_Skin_FFPE' + '.h5ad'

adata = anndata.read(sample_name)

data_list=defaultdict(list)
for i in range(0, len(adata.obsm['spatial'])):
    if adata.obsm['spatial'][i][0]<6500:
        continue
    data_list['x_axis'].append(adata.obsm['spatial'][i][0])
    data_list['y_axis'].append(adata.obsm['spatial'][i][1])

data_list_pd = pd.DataFrame(data_list)
chart = alt.Chart(data_list_pd).mark_point(filled=True,size=0.7).encode(
    alt.X('x_axis', scale=alt.Scale(zero=False)),
    alt.Y('y_axis', scale=alt.Scale(zero=False))
    #color=alt.Color('cluster_label:N', scale=alt.Scale(range=colors))
)#.configure_legend(labelFontSize=6, symbolLimit=50)

chart.save(output_name+'_tissue_plot.html')


