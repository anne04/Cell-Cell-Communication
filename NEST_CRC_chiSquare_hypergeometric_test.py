print('package loading')
import numpy as np
import csv
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import  rgb2hex # LinearSegmentedColormap, to_hex,
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
import gzip
import argparse
import os
import scipy.stats
from scipy.sparse.csgraph import connected_components
from pyvis.network import Network
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import altair as alt
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
import gc
import copy
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")


#current_directory = ??

##########################################################
# preprocessDf, plot: these two functions are taken from GW's repository                                                                                                                                                                     /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/fatema_signaling/hist.py                                                                                                                                                                                         

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

####################### Set the name of the sample you want to visualize ###################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( '--data_name', type=str, help='The name of dataset', default="Visium_HD_Human_Colon_Cancer_square_002um_outputs") # , required=True
    parser.add_argument( '--model_name', type=str, help='Name of the trained model', default='NEST_Visium_HD_Human_Colon_Cancer_square_002um_outputs') #, required=True
    parser.add_argument( '--top_edge_count', type=int, default=135000 ,help='Number of the top communications to plot. To plot all insert -1') # 
    parser.add_argument( '--top_percent', type=int, default=20, help='Top N percentage communications to pick')    
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--output_path', type=str, default='output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--barcode_info_file', type=str, default='', help='Path to load the barcode information file produced during data preprocessing step')
    parser.add_argument( '--annotation_file_path', type=str, default='', help='Path to load the annotation file in csv format (if available) ')
    parser.add_argument( '--selfloop_info_file', type=str, default='', help='Path to load the selfloop information file produced during data preprocessing step')
    parser.add_argument( '--top_ccc_file', type=str, default='', help='Path to load the selected top CCC file produced during data postprocessing step')
    parser.add_argument( '--output_name', type=str, default='', help='Output file name prefix according to user\'s choice')
    parser.add_argument( '--filter', type=int, default=1, help='Set --filter=-1 if you want to filter the CCC')
    parser.add_argument( '--filter_by_ligand_receptor', type=str, default='', help='Set ligand-receptor pair, e.g., --filter_by_ligand_receptor="CCL19-CCR7" if you want to filter the CCC by LR pair')
    parser.add_argument( '--filter_by_annotation', type=str, default='', help='Set cell or spot type, e.g., --filter_by_annotation="T-cell" if you want to filter the CCC')
    parser.add_argument( '--filter_by_component', type=int, default=32, help='Set component id, e.g., --filter_by_component=9 if you want to filter by component id')
    
    
    args = parser.parse_args()
    if args.metadata_from=='metadata/': # if default one is used, then concatenate the dataname. Otherwise, use the user provided path directly
        args.metadata_from = args.metadata_from + args.data_name + '/'

    if args.output_path=='output/': # if default one is used, then concatenate the dataname. Otherwise, use the user provided path directly
        args.output_path = args.output_path + args.data_name + '/'
    print('Top %d communications will be plot. To change the count use --top_edge_count parameter'%args.top_edge_count)

    if args.output_name=='':
        output_name = args.output_path + args.model_name
    else: 
        output_name = args.output_path + args.output_name
    
    ##################### make cell metadata: barcode_info ###################################
    if args.barcode_info_file=='':
        with gzip.open(args.metadata_from +args.data_name+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   
            barcode_info = pickle.load(fp)
    else:
        with gzip.open(args.barcode_info_file, 'rb') as fp:  #b, a:[0:5]        
            barcode_info = pickle.load(fp)    


    ###############################  read which spots have self loops ################################################################
    if args.selfloop_info_file=='':
        with gzip.open(args.metadata_from + args.data_name +'_self_loop_record', 'rb') as fp:  #b, a:[0:5]   _filtered
            self_loop_found = pickle.load(fp)
    else:
        with gzip.open(args.selfloop_info_file, 'rb') as fp:  #b, a:[0:5]   _filtered
            self_loop_found = pickle.load(fp)

    ####### load annotations ##############################################
    if args.annotation_file_path != '':
        pathologist_label=[]
        annotation_data = pd.read_csv(args.annotation_file_path, sep=",")
        for i in range (0, len(annotation_data)):
            pathologist_label.append([annotation_data['Barcode'][i], annotation_data['Type'][i]])

        barcode_type=dict() # record the type (annotation) of each spot (barcode)
        for i in range (0, len(pathologist_label)):
            barcode_type[pathologist_label[i][0]] = pathologist_label[i][1]


    else:
        barcode_type=dict() # record the type (annotation) of each spot (barcode)
        for i in range (0, len(barcode_info)):
            barcode_type[barcode_info[i][0]] = ''

    ######################### read the NEST output in csv format ####################################################
    if args.top_ccc_file == '':
        inFile = args.output_path + args.model_name+'_top' + str(args.top_percent) + 'percent.csv'
        df = pd.read_csv(inFile, sep=",")
    else: 
        inFile = args.top_ccc_file
        df = pd.read_csv(inFile, sep=",")


    csv_record = df.values.tolist() # barcode_info[i][0], barcode_info[j][0], ligand, receptor, edge_rank, label, i, j, score

    ## sort the edges based on their rank (column 4), low to high, low being higher attention score
    csv_record = sorted(csv_record, key = lambda x: x[4])
    ## add the column names and take first top_edge_count edges
    # columns are: from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score 
    df_column_names = list(df.columns)
#    print(df_column_names)

    print(len(csv_record))

    if args.top_edge_count != -1:
        csv_record_final = [df_column_names] + csv_record[0:min(args.top_edge_count, len(csv_record))]

    ## add a dummy row at the end for the convenience of histogram preparation (to keep the color same as altair plot)
    in_region_node = -1
    for i in range (0, len(barcode_info)):
        if barcode_info[i][1] <= 54000 :
            in_region_node = i
            break
  
    i = in_region_node
    j = in_region_node
    csv_record_final.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j, 0]) # dummy for histogram

    csv_record = 0
    gc.collect()

    ######################## connected component finding #################################
    print('Finding connected component')
    connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
    for k in range (1, len(csv_record_final)-1): # last record is a dummy for histogram preparation
        i = csv_record_final[k][6]
        j = csv_record_final[k][7]
        connecting_edges[i][j]=1
            
    graph = csr_matrix(connecting_edges)
    n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) # It assigns each SPOT to a component based on what pair it belongs to
    print('Number of connected components %d'%n_components) 

    count_points_component = np.zeros((n_components))
    for i in range (0, len(labels)):
        count_points_component[labels[i]] = count_points_component[labels[i]] + 1

    id_label = 2 # initially all are zero. =1 those who have self edge but above threshold. >= 2 who belong to some component
    index_dict = dict()
    for i in range (0, count_points_component.shape[0]):
        if count_points_component[i]>1:
            index_dict[i] = id_label
            id_label = id_label+1

    print('Unique component count %d'%id_label)

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
    
    ############################################### Optional filtering ########################################################
    if args.filter == 1:
        ## change the csv_record_final here if you want histogram for specific components/regions only. e.g., if you want to plot only stroma region, or tumor-stroma regions etc.    ##
        csv_record_final_temp = []
        csv_record_final_temp.append(csv_record_final[0])
        ligand_receptor_pair = defaultdict(list)
        for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
            if args.filter_by_component!=-1:
                if csv_record_final[record_idx][5] == int(args.filter_by_component):
                    csv_record_final_temp.append(csv_record_final[record_idx])
                    if csv_record_final[record_idx][2]=='APP' and (csv_record_final[record_idx][3]=='ITGA6' or csv_record_final[record_idx][3]=='TGFBR2'):
                        pair = csv_record_final[record_idx][2] + "-" + 'ITGA6/TGFBR2'
                    else:  
                        pair = csv_record_final[record_idx][2] + "-" + csv_record_final[record_idx][3]
                    ligand_receptor_pair[pair].append('')                
        
        csv_record_final_temp.append(csv_record_final[len(csv_record_final)-1])
        csv_record_final = copy.deepcopy(csv_record_final_temp)

    ######## preprocessing for chi-square and hypergeometric test ####################################################
    total_count = len(csv_record_final_temp)-1
    total_type = len(list(ligand_receptor_pair.keys()))

    f_obs = []
    position_target = 0
    count = 0
    occurance_percentage = [] 
    for pair in ligand_receptor_pair.keys():
        f_obs.append(len(ligand_receptor_pair[pair]))
        occurance_percentage.append(len(ligand_receptor_pair[pair])/total_count)
        if pair=='APP-ITGA6/TGFBR2': #or pair=='APP-TGFBR2':
            position_target = count
            print('position_target %d'%position_target)
        count = count+1

    sample_size = 100
    for i in range (0, len(occurance_percentage)):
        occurance_percentage[i] = int(np.round(occurance_percentage[i] * sample_size))

    # due to round down, total may be less that 100. So adjust the target LR pair to have more occurance, so that the total will be 100
    occurance_percentage[position_target] = occurance_percentage[position_target] +  (sample_size-int(np.sum(occurance_percentage)))
    print('type, x(=how many selected from the type), m(=total count from the type)')
    for i in range (0, len(occurance_percentage)):
        print('%d, %d, %d'%(i, occurance_percentage[i], f_obs[i]))


    ######## Hypergeometric test ####################################################

    from scipy.stats import multivariate_hypergeom

    m_data = f_obs # actual observation count for each lr-pair (variable)
    n_data = sample_size # total draw = 100
    x_data = occurance_percentage # expected count of draw from each lr-pair. For PLXNB2-MET = 20%  
    # Null hypothesis: Out of 100 draw, only PLXNB2-MET is chosen 20% of the time (the rest 80% are distributed among the rest 217 pairs), just by chance. 
    # Alternative hypothesis: Out of 100 draw, only APP-ITGA6/TGFBR2 is chosen 20% of the time NOT by chance, but because it is biased. 
    print('hypergeometric probability of null hypothesis: APP-ITGA6/TGFBR2 wil be selected most of the time out of %d draws just by chance is: %g' % (n_data, multivariate_hypergeom.pmf(x=x_data, m=m_data, n=n_data)))
    # p-value < 0.05 -- so reject the null hypothesis and accept the alternative hypothesis.
    # the null hypothesis that there is nothing special about the jar. If this probability (also called the p-value) is sufficiently low, then we can decide to reject the null hypothesis as too unlikely 
    # — something must be going on with this jar.    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_hypergeom.html

    ######## chi-square test ####################################################
    degree_of_freedom = total_type - 1
    f_obs = np.array(f_obs)
    chisqr = scipy.stats.chisquare(f_obs)   
    # Null hypothesis: All lr-pairs occur the equal number of times.
    # Alternative hypothesis: Some lr-pairs are occurring significantly more number of times than the rest. So it is skewed. 
    print('total_count %d, total_type %d, degree of freedom %d, chi square test statistic p-value of all having equal probability of occurance = %g '%(total_count, total_type, degree_of_freedom, chisqr[1])) 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html   
    # p-value < 0.05 -- so reject the null hypothesis and accept the alternative hypothesis.
