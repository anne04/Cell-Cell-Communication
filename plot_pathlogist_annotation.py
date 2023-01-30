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


def main(args):
    print(args.data_path)
    adata_h5 = st.Read10X(path=data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
    print('filtering genes based on min_cell = 1')
    sc.pp.filter_genes(adata_h5, min_cells=1)
    #print(adata_h5)
    gene_ids = list(adata_h5.var_names)
    cell_coordinates = adata_h5.obsm['spatial']
    cell_coordinates = np.array(cell_coordinates)
    cell_barcode = list(adata_h5.obs.index)

    # combine cell barcodes and coordinates into one data structure
    barcode_info=[]
    i=0
    with open(barcode_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            barcode_info.append([cell_barcode[0], cell_coordinates[i,0], cell_coordinates[i,1], 0]) # last position of each entry will hold the label
            i=i+1

    ############
    pathologist_label_file= args.pathologist_label # tumor_64630_D1_IX_annotation.csv'
    pathologist_label=[]
    with open(pathologist_label_file) as file:
        csv_file = csv.reader(file, delimiter=",")
        for line in csv_file:
            pathologist_label.append(line)

    barcode_label=dict()
    count=np.zeros((4))
    for i in range (1, len(pathologist_label)):
      if pathologist_label[i][1] == 'tumor': #'Tumour':
          barcode_label[pathologist_label[i][0]] = 1
          count[0] = count[0] + 1
      elif pathologist_label[i][1] == 'stroma_deserted': #'Stroma':
          barcode_label[pathologist_label[i][0]] = 2
          count[1] = count[1] + 1
      elif pathologist_label[i][1] == 'acinar_reactive': #'Acinar_reactive':  
          barcode_label[pathologist_label[i][0]] = 3
          count[2] = count[2] + 1
      elif pathologist_label[i][1] == 'Artifact':  
          barcode_label[pathologist_label[i][0]] = 4
          count[3] = count[3] + 1

    max = 4
    ############
    for i in range (0, len(barcode_info)):
        if barcode_info[i][0] in barcode_label:
            barcode_info[i][3] = barcode_label[barcode_info[i][0]]

    #count=0      
    for label_i in range (0, max+1):
        x_index=[]
        y_index=[]
        for i in range (0, len(barcode_info)):
            if barcode_info[i][3] == label_i:
                x_index.append(barcode_info[i][1])
                y_index.append(barcode_info[i][2])
                #if label_i==0:
                #    count=count+1


        plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = label_i)

    #print(count)
    plt.legend()


    plt.savefig(args.save_path+'pathologists_plot_new.png', dpi=400)
    plt.clf()
    # 413 == barcode not found, 443 = not labeled

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/', help='The path to dataset')
    parser.add_argument( '--save_path', type=str, default='/cluster/home/t116508uhn/64630/', help='The name of dataset')
    parser.add_argument( '--pathologist_label', type=str, default='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv', help='The name of dataset')
    
    
    args = parser.parse_args()
    main(args)
    
    
