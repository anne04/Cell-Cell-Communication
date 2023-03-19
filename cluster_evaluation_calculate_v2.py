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
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, help='The name of dataset')
    parser.add_argument( '--cluster_label_file', type=str, default='kmeans_gene_exp_barcode_label.csv', help='cluster file?')
    parser.add_argument( '--true_label_file', type=str, help='true label file?')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--result_path', type=str, default='result/')

    args = parser.parse_args()
    
    generated_data_fold = args.generated_data_path + args.data_name+'/'
    result_path = args.result_path +'/'+ args.data_name +'/'
    #cell_label_file= result_path+args.cell_label_file_name
    save_path = result_path
    
    #coordinates = np.load(generated_data_fold+'coordinates.npy')
    cell_barcode = np.load(generated_data_fold+'barcodes.npy', allow_pickle=True)
    barcode_label=[]
    for i in range (0, len(cell_barcode)):
        barcode_label.append(cell_barcode[i])

     
#################################################################################               
    toomany_label_file=args.cluster_label_file
    toomany_label=[]
    with open(toomany_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            toomany_label.append(line)

    barcode_label_pred=dict()
    cluster_dict=defaultdict(list)
    for i in range (1, len(toomany_label)):
        if len(toomany_label[i])>0 :
            barcode_label_pred[toomany_label[i][0]] = int(toomany_label[i][1])
            cluster_dict[int(toomany_label[i][1])].append(toomany_label[i][0])

    print('total number of predicted clusters: %d '%len(cluster_dict.keys()))

#################################################################################       
    pathologist_label_file=args.true_label_file
    pathologist_label=dict()
    with open(pathologist_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            pathologist_label[line[1]].append(line[0]) # it means: pathologist_label[cluster_id].append(barcode)

                
    barcode_label_pathologist=dict()
    #count=np.zeros((4))
    true_label_dict = defaultdict(list)
    cluster_id_int = 0
    for cluster_id in pathologist_label: # cluster_id is string. We need int format. 
        for barcode in pathologist_label[cluster_id]:
            barcode_label_pathologist[barcode] = cluster_id_int 
            true_label_dict[cluster_id_int].append(barcode)         
        cluster_id_int = cluster_id_int + 1

###################### Homogeneity and ARI ###########################################################  

    # node vs path
    spot_node_pred = []
    spot_real = []
    barcode_keys=list(barcode_label_pred.keys())
    for barcode in barcode_keys:
        if barcode in barcode_label_pathologist:
            spot_real.append(barcode_label_pathologist[barcode])
            spot_node_pred.append(barcode_label_pred[barcode])


    #print(normalized_mutual_info_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.10
    print('Homogeneity: pathologist: %g '%homogeneity_score(labels_true=spot_real,labels_pred=spot_node_pred)) # pred vs pathologist: 0.33
    print('ARI: pathologist: %g '%adjusted_rand_score(labels_true=spot_real, labels_pred=spot_node_pred) )


############### purity ##############

    max_matched = np.zeros((len(cluster_dict.keys())))
    cluster_list = list(cluster_dict.keys())

    for k in range (0, len(cluster_list)):
        p = cluster_list[k]
        temp_max = 0
        for t in true_label_dict.keys():  
            count = 0
            barcodes_list = true_label_dict[t]
            for barcode in barcodes_list:
                if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
                    if barcode in cluster_dict[p]:
                        count = count+1

            if count > temp_max: 
                temp_max = count

        max_matched[k] = temp_max

    N_cells = 0
    for barcode in barcode_label:
        if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
            N_cells = N_cells + 1

    purity_cluster = np.sum(max_matched)/N_cells 
    print('purity: pathologist: %g'%purity_cluster)       

    ##############  entropy ###############

    entropy_cluster = np.zeros((len(cluster_dict.keys())))
    cluster_list = list(cluster_dict.keys())


    for k in range (0, len(cluster_list)):
        p = cluster_list[k]
        cluster_count = len(cluster_dict[p])

        H_sum = 0
        for t in true_label_dict.keys():  
            count_match = 0
            barcodes_list = true_label_dict[t]
            for barcode in barcodes_list:
                if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
                    if barcode in cluster_dict[p]:
                        count_match = count_match+1

            #print(count_match/cluster_count)
            if count_match/cluster_count != 0:
                H_sum = H_sum + (count_match/cluster_count)*np.log(count_match/cluster_count)


        entropy_cluster[k] = H_sum


    N_cells = 0
    for barcode in barcode_label:
        if barcode in barcode_label_pathologist and barcode in barcode_label_pred: 
            N_cells = N_cells + 1



    entropy_total = 0
    for k in range (0, len(cluster_list)):
        entropy_total = entropy_total + (len(cluster_dict[cluster_list[k]])*entropy_cluster[k])/N_cells

    entropy_total = - entropy_total
    print('entropy_total: pathologist: %g'%entropy_total)           
