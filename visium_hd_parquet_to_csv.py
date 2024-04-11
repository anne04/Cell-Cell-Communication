import pandas as pd
df_coordinate = pd.read_parquet('/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_positions.parquet')
df.to_csv('/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_positions_list.csv',header=None, index=None)
