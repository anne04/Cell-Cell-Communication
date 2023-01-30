import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List

import altair as alt
from vega_datasets import data
import pandas as pd

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, default='64630_Spatial10x', help='The name of dataset')
    parser.add_argument( '--cell_label_file', type=str, default='kmeans_gene_exp_barcode_label.csv', help='data name?')
    parser.add_argument( '--generated_data_path', type=str, default='/project/def-gregorys/fatema/GCN_clustering/generated_data/', help='data path')
    parser.add_argument( '--result_path', type=str, default='/project/def-gregorys/fatema/GCN_clustering/result/')

    args = parser.parse_args()
    
    generated_data_fold = args.generated_data_path + args.data_name+'/'
    result_path = args.result_path +'/'+ args.data_name +'/'
    cell_label_file= args.generated_data_path + args.data_name+'/'+args.cell_label_file_name
    save_path = result_path
    
    coordinates = np.load(generated_data_fold+'coordinates.npy')
    cell_barcode = np.load(generated_data_fold+'barcode.npy', allow_pickle=True)
    barcode_info=[]

    for i in range (0, coordinates.shape[0]):
        barcode_info.append([cell_barcode[i], coordinates[i,0],coordinates[i,1],0])



    ############################################################################################################

    number = 20
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    number = 20
    cmap = plt.get_cmap('tab20b')
    colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

    colors=colors+colors_2

    number = 20
    cmap = plt.get_cmap('tab20c')
    colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

    colors=colors+colors_2

    number = 8
    cmap = plt.get_cmap('Set2')
    colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

    colors=colors+colors_2

    number = 12
    cmap = plt.get_cmap('Set3')
    colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

    colors=colors+colors_2

    for i in range (0, len(colors)): 
        colors[i] = matplotlib.colors.to_hex([colors[i][0], colors[i][1], colors[i][2], colors[i][3]])

    #####



    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label_node_embedding.csv'
    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label_node_embedding.csv'
    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label_node_embedding.csv'

    #toomany_label_file='/cluster/home/t116508uhn/64630/TAGConv_test_r4_too-many-cell-clusters_org.csv'
    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/leiden_barcode_label.csv'
    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/louvain_barcode_label.csv'
    #toomany_label_file='new_alignment/result_lp8mp2_bulk/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/kmeans_barcode_label.csv'
    #toomany_label_file='/cluster/home/t116508uhn/64630/GCN_r7_toomanycells_minsize20_labels.csv'
    #toomany_label_file='/cluster/home/t116508uhn/64630/PCA_64embedding_pathologist_label_l1mp5_temp.csv' #'/cluster/home/t116508uhn/64630/PCA_64embedding_Kena_label_l1mp5_temp.csv'
    #toomany_label_file='/cluster/home/t116508uhn/64630/spaceranger_pathologist.csv'
    cell_label=[]
    with open(cell_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            cell_label.append(line)
            
    barcode_label=dict()
    for i in range (1, len(cell_label)):
        if len(cell_label[i])>0 :
            barcode_label[cell_label[i][0]] = int(cell_label[i][1])


    for i in range (0, len(barcode_info)):
        if barcode_info[i][0] in barcode_label:
            barcode_info[i][3] = barcode_label[barcode_info[i][0]]


    data_list=dict()
    data_list['cluster_label']=[]
    data_list['X']=[]
    data_list['Y']=[]

    for i in range (0, len(barcode_info)):
        data_list['cluster_label'].append(barcode_info[i][3])
        data_list['X'].append(barcode_info[i][1])
        data_list['Y'].append(-barcode_info[i][2])

    data_list_pd = pd.DataFrame(data_list)
    #######


    chart = alt.Chart(data_list_pd).mark_point(filled=True).encode(
        alt.X('X', scale=alt.Scale(zero=False)),
        alt.Y('Y', scale=alt.Scale(zero=False)),
        #alt.Size('pop:Q'),
        color=alt.Color('cluster_label:N', scale=alt.Scale(range=colors))
    ).configure_legend(labelFontSize=6, symbolLimit=50)



    chart.save(save_path+'cluster_plot.html')



