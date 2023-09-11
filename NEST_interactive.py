import numpy as np
import csv
import pickle
from scipy import sparse
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
import matplotlib
#from typing import List
import qnorm
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
import gzip
#import copy 
import argparse
import os
from scipy.sparse.csgraph import connected_components
import scipy.stats
import copy
import altairThemes
import altair as alt

alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")


############# preprocessDf, plot: these three functions are taken from GW's repository ####################################################################################                                                                                                                                                                  /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/fatema_signaling/hist.py                                                                                                                                                                                         

def preprocessDf(df):
  """Transform ligand and receptor columns."""
  df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
  df["component"] = df["component"] #.astype(str).str.zfill(2)
  return df

def plot(df):
  set1 = altairThemes.get_colour_scheme("Set1", len(df["component"].unique()))
  set1[0] = '#000000'
  base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale = alt.Scale(range=set1)),
            order=alt.Order("component:N", sort="ascending"),
            tooltip=["component"]
        )
  p = base
  return p

#########################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/'  , help='The path to dataset') 
    parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
    parser.add_argument( '--top_edge_count', type=int, default=1000, help='How many top attention scored edges you want to plot')
    args = parser.parse_args()
    
    
    ####################### Set the name of the sample you want to visualize #####################################
    current_directory = '/cluster/home/t116508uhn/64630/' # args.data_path
    data_name = 'PDAC_64630' #args.data_name #'LUAD_GSM5702473_TD1' #LUAD_GSM5702473_TD1
    top_edge_count = 1300 # args.top_edge_count # how many top edges you want to visualize
    
    ######################## read data in csv format ########################################
    gene_ids = pd.read_csv(current_directory+'gene_ids_'+args.data_name+'.csv', header=None)
    gene_ids = list(gene_ids[0]) # first column is the gene ids. Convert it to list format for conveniance
    
    cell_barcode = pd.read_csv(current_directory+'cell_barcode_'+args.data_name+'.csv', header=None)
    cell_barcode = list(cell_barcode[0]) # first column is: cell barcode. Convert it to list format for conveniance
    
    temp = pd.read_csv(current_directory+'coordinates_'+args.data_name+'.csv', header=None)
    coordinates = np.zeros((len(cell_barcode), 2)) # num_cells x coordinate
    for i in range (0, len(temp)):
        coordinates[i,0] = temp[0][i] # x
        coordinates[i,1] = temp[1][i] # y
    
    with gzip.open(current_directory+'self_loop_record_'+args.data_name, 'rb') as fp:  #b, a:[0:5]   _filtered
    	self_loop_found = pickle.load(fp)
        
    ##################### make cell metadata: barcode_info ###################################
    
    i=0
    barcode_serial = dict()
    for cell_code in cell_barcode:
        barcode_serial[cell_code]=i
        i=i+1
        
    i=0
    barcode_info=[]
    for cell_code in cell_barcode:
        barcode_info.append([cell_code, coordinates[i,0],coordinates[i,1], 0]) # last entry will hold the component number later
        i=i+1
    
    ####### load annotations ##############################################
    pathologist_label_file='/cluster/home/t116508uhn/IX_annotation_artifacts.csv' 
    pathologist_label=[]
    with open(pathologist_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            pathologist_label.append(line)	
        
    barcode_type=dict() # record the type (annotation) of each spot (barcode)
    for i in range (1, len(pathologist_label)):
        barcode_type[pathologist_label[i][0]] = pathologist_label[i][1]
        
    ######################### read the NEST output in csv format ####################################################
    
    filename_str = 'NEST_combined_rank_product_output_'+args.data_name+'_top20percent.csv'
    inFile = current_directory +filename_str 
    df = pd.read_csv(inFile, sep=",")
    
    print('All reading done. Now start processing.')
    
    ######################## sort the edges in ascending order of rank and take top_edge_count edges to plot ##########
    
    csv_record = df.values.tolist()
    
    ## sort the edges based on their rank (column 4) column, low to high, low rank being higher attention score
    csv_record = sorted(csv_record, key = lambda x: x[4])
    
    ## add the column names and take first top_edge_count edges
    # columns are: from_cell, to_cell, ligand_gene, receptor_gene, attention_score, component, from_id, to_id
    df_column_names = list(df.columns)
    csv_record_final = [df_column_names] + csv_record[0:top_edge_count]
    
    ## add a dummy row at the end for the convenience of histogram preparation (to keep the color same as altair plot)
    i=0
    j=0
    csv_record_final.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j]) # dummy for histogram
    
    ############################################### Optional filtering ########################################################
    '''
    ## change the csv_record_final here if you want histogram for specific components/regions only. e.g., if you want to plot only stroma region, or tumor-stroma regions etc.    ##
    #region_of_interest = [...] 
    csv_record_final_temp = []
    csv_record_final_temp.append(csv_record_final[0])
    for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
        # if at least one spot of the pair is tumor, then plot it
        if (barcode_type[csv_record_final[record_idx][0]] == 'tumor' or barcode_type[csv_record_final[record_idx][1]] == 'tumor'): #((barcode_type[csv_record_final[record_idx][0]] == 'tumor' and barcode_type[csv_record_final[record_idx][1]] == 'tumor') or (barcode_type[csv_record_final[record_idx][0]] != 'tumor' and barcode_type[csv_record_final[record_idx][1]] != 'tumor')):
            csv_record_final_temp.append(csv_record_final[record_idx])
            
    csv_record_final_temp.append(csv_record_final[len(csv_record_final)-1])
    csv_record_final = copy.deepcopy(csv_record_final_temp)
    '''
    
    ######################## connected component finding #################################
    
    connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
    for k in range (1, len(csv_record_final)-1): # last record is a dummy for histogram preparation
        i = csv_record_final[k][6]
        j = csv_record_final[k][7]
        connecting_edges[i][j]=1
            
    graph = csr_matrix(connecting_edges)
    n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) # It assigns each SPOT to a component based on what pair it belongs to
    print('number of connected components %d'%n_components) 
    
    count_points_component = np.zeros((n_components))
    for i in range (0, len(labels)):
         count_points_component[labels[i]] = count_points_component[labels[i]] + 1
    
    print(count_points_component)
    
    id_label = 2 # initially all are zero. = 1 those who have self edge. >= 2 who belong to some component
    index_dict = dict()
    for i in range (0, count_points_component.shape[0]):
        if count_points_component[i]>1:
            index_dict[i] = id_label
            id_label = id_label+1
    
    print(id_label)
    
    for i in range (0, len(barcode_info)):
        if count_points_component[labels[i]] > 1:
            barcode_info[i][3] = index_dict[labels[i]] #2
        elif connecting_edges[i][i] == 1 and (i in self_loop_found and i in self_loop_found[i]): # that is: self_loop_found[i][i] do exist 
            barcode_info[i][3] = 1
        else: 
            barcode_info[i][3] = 0
    
    # update the label based on found component numbers
    #max opacity
    for record in range (1, len(csv_record_final)-1):
        i = csv_record_final[record][6]
        label = barcode_info[i][3]
        csv_record_final[record][5] = label
    
    
    component_list = dict()
    for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
        record = csv_record_final[record_idx]
        i = record[6]
        j = record[7]
        component_label = record[5]
        barcode_info[i][3] = component_label 
        barcode_info[j][3] = component_label
        component_list[component_label] = ''
    
    component_list[0] = ''
    unique_component_count = len(component_list.keys())
    
    
    ##################################### Altair Plot ##################################################################
    ## dictionary of those spots who are participating in CCC ##
    active_spot = defaultdict(list)
    for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
        record = csv_record_final[record_idx]
        i = record[6]
        pathology_label = barcode_type[barcode_info[i][0]]
        component_label = record[5]
        X = barcode_info[i][1]
        Y = -barcode_info[i][2]
        opacity = np.float(record[8])
        active_spot[i].append([pathology_label, component_label, X, Y, opacity])
        
        j = record[7]
        pathology_label = barcode_type[barcode_info[j][0]]
        component_label = record[5]
        X = barcode_info[j][1]
        Y = -barcode_info[j][2]
        opacity = np.float(record[8])   
        active_spot[j].append([pathology_label, component_label, X, Y, opacity])
        ''''''
        
    ######### color the spots in the plot with opacity = attention score #################
    opacity_list = []
    for i in active_spot:
        sum_opacity = []
        for edges in active_spot[i]:
            sum_opacity.append(edges[4])
            
        avg_opacity = np.max(sum_opacity) #np.mean(sum_opacity)
        opacity_list.append(avg_opacity) 
        active_spot[i]=[active_spot[i][0][0], active_spot[i][0][1], active_spot[i][0][2], active_spot[i][0][3], avg_opacity]
    
    
    #### making dictionary for converting to pandas dataframe to draw altair plot ###########
    data_list=dict()
    data_list['pathology_label']=[]
    data_list['component_label']=[]
    data_list['X']=[]
    data_list['Y']=[]   
    data_list['opacity']=[]  
    
    for i in range (0, len(barcode_info)):        
        if i in active_spot:
            data_list['pathology_label'].append(active_spot[i][0])
            data_list['component_label'].append(active_spot[i][1])
            data_list['X'].append(active_spot[i][2])
            data_list['Y'].append(active_spot[i][3])
            data_list['opacity'].append(active_spot[i][4])
            
        else:
            data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
            data_list['component_label'].append(0) # make it zero so it is black
            data_list['X'].append(barcode_info[i][1])
            data_list['Y'].append(-barcode_info[i][2])
            data_list['opacity'].append(0.1)
    
    
    
    # converting to pandas dataframe
    
    data_list_pd = pd.DataFrame(data_list)
    id_label = len(list(set(data_list['component_label'])))#unique_component_count
    set1 = altairThemes.get_colour_scheme("Set1", id_label)
    set1[0] = '#000000'
    chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
        alt.X('X', scale=alt.Scale(zero=False)),
        alt.Y('Y', scale=alt.Scale(zero=False)),
        shape = alt.Shape('pathology_label:N'), 
        color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
        #opacity=alt.Opacity('opacity:N'), #"opacity", # ignore the opacity for now
        tooltip=['component_label']  
    )
    
    chart.save(current_directory +'altair_plot_test.html')
    ###################################  Histogram plotting #################################################################################
    
    df = pd.DataFrame(csv_record_final)
    df.to_csv(current_directory+'temp_csv.csv', index=False, header=False)
    df =  pd.read_csv(current_directory+'temp_csv.csv', sep=",")
    os.remove(current_directory+'temp_csv.csv') # delete the intermediate file
    df = preprocessDf(df)
    p = plot(df)
    outPath = current_directory+'histogram_test.html'
    p.save(outPath)	
    
    
    ############################  Network Plot ######################
    import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
    set1 = altairThemes.get_colour_scheme("Set1", unique_component_count)
    colors = set1
    colors[0] = '#000000'
    ids = []
    x_index=[]
    y_index=[]
    colors_point = []
    for i in range (0, len(barcode_info)):    
        ids.append(i)
        x_index.append(barcode_info[i][1])
        y_index.append(barcode_info[i][2])    
        colors_point.append(colors[barcode_info[i][3]]) 
      
    max_x = np.max(x_index)
    max_y = np.max(y_index)
    
    from pyvis.network import Network
    import networkx as nx
    
    
    barcode_type=dict()
    for i in range (1, len(pathologist_label)):
        if 'tumor'in pathologist_label[i][1]: #'Tumour':
            barcode_type[pathologist_label[i][0]] = 1
        else:
            barcode_type[pathologist_label[i][0]] = 0
    
    g = nx.MultiDiGraph(directed=True) #nx.Graph()
    for i in range (0, len(barcode_info)):
        label_str =  str(i)+'_c:'+str(barcode_info[i][3])+'_' # label of the node or spot is consists of: spot id, component number, type of the spot 
        if barcode_type[barcode_info[i][0]] == 0: #stroma
            marker_size = 'circle'
            label_str = label_str + 'stroma'
        elif barcode_type[barcode_info[i][0]] == 1: #tumor
            marker_size = 'box'
            label_str = label_str + 'tumor'
        else:
            marker_size = 'ellipse'
            label_str = label_str + 'acinar_reactive'
    	
        g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(-y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    
    
    
    count_edges = 0
    for k in range (1, len(csv_record_final)-1):
        i = csv_record_final[k][6]
        j = csv_record_final[k][7]    
        ligand = csv_record_final[k][2]
        receptor = csv_record_final[k][3]
        edge_score = csv_record_final[k][8] 
        title_str =  "L:" + ligand + ", R:" + receptor+ ", "+ str(edge_score) #+
        g.add_edge(int(i), int(j), label = title_str, color=colors_point[i], value=np.float64(edge_score)) #
        count_edges = count_edges + 1
    
    print("total edges plotted: %d"%count_edges)
    
    nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True     
    nt.from_nx(g)
    nt.save_graph('mygraph.html')
    os.system('cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html')
    
    ################## The rest can be ignored ##############################################################################
    
    # convert it to dot file to be able to convert it to pdf or svg format for inserting into the paper
    from networkx.drawing.nx_agraph import write_dot
    write_dot(g, "/cluster/home/t116508uhn/64630/test_interactive.dot")

'''
#These commands are to be executed in the linux terminal to convert the .dot file to pdf/svg:
cat test_interactive.dot.dot  | sed 's/ellipse/triangle/g'   | sed 's/tumor",/tumor",style="filled",/g'   | sed 's/L:\([^ ]\+\), R:/\1-/g'   | sed 's/label="[0-9][^"]*"/label=""/g' | awk -F'=' '{ if ($1 == "penwidth") {print $1 "=" ($2 ^ 6) ","} else {print $0 }}'   | tr '\n' ' '   | sed "s/;/\n/g"  > tmp
cat tmp   | dot -Kneato -n -y -Tpdf -Efontname="Arial" -Nlabel="" -Nwidth=1.5 -Nheight=1.5 -Npenwidth=8	> test.pdf
cat tmp   | dot -Kneato -n -y -Tsvg -Efontname="Arial" -Nlabel="" -Nwidth=1.5 -Nheight=1.5 -Npenwidth=8	> test.svg
'''
####### Read the raw data to get the gene id, cell barcode, cell coordinates, cell vs gene expression matrix ##########################################################################
'''
import stlearn as st
adata_h5 = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
sc.pp.filter_genes(adata_h5, min_cells=1)

gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
cell_barcode = np.array(adata_h5.obs.index)

temp = adata_h5.X
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(temp)))  
cell_vs_gene = np.transpose(temp)  

######################## save the gene ids, cell_carcodes, coordinates, cell_vs_gene matrix ######################
df = pd.DataFrame(gene_ids)
df.to_csv(current_directory+'gene_ids_'+args.data_name+'.csv', index=False, header=False)

df = pd.DataFrame(coordinates)
df.to_csv(current_directory+'coordinates_'+args.data_name+'.csv', index=False, header=False)

df = pd.DataFrame(cell_barcode)
df.to_csv(current_directory+'cell_barcode_'+args.data_name+'.csv', index=False, header=False)

with gzip.open(current_directory+'cell_vs_gene_quantile_transformed_'+args.data_name, 'wb') as fp:  #b, a:[0:5]   _filtered
	pickle.dump(cell_vs_gene, fp)
'''
