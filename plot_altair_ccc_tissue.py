#sys.path.append("/path/to/parent/directory/of/altairThemes.py")

import pandas as pd
import matplotlib
import altair as alt
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.

# register the custom theme under a chosen name
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")



data_list_pd = pd.read_csv('/cluster/home/t116508uhn/64630/ccc_th95_tissue_plot.csv')

set1 = altairThemes.get_colour_scheme("Set1", len(data_list_pd["component_label"].unique()))
    
chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
    alt.X('X', scale=alt.Scale(zero=False), axis=alt.Axis(grid=False)),
    alt.Y('Y', scale=alt.Scale(zero=False), axis=alt.Axis(grid=False)),
    shape = "pathology_label",
    color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
    tooltip=['component_label']
)#.configure_legend(labelFontSize=6, symbolLimit=50)

save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.html')
