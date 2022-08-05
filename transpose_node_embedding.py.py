import numpy as np
import csv
#X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
X_embedding_filename = '/cluster/projects/schwartzgroup/fatema/CCST/Embedding_data_NoPCA_512_quantile/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/lambdaI0.3_epoch10000_Embed_X.npy'
X_embedding = np.load(X_embedding_filename)

X_embedding_T = np.transpose(X_embedding)
X_embedding_filename = '/cluster/home/t116508uhn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_Embed_X.csv'
np.savetxt(X_embedding_filename, X_embedding_T, delimiter=",")


import csv
barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
barcode_info=[]
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append(line)
        
barcode_label_file='/cluster/home/t116508uhn/64630/node_label.csv'
barcode_label_info=[]
with open(barcode_label_file) as file:
    tsv_file = csv.reader(file, delimiter=",")
    for line in tsv_file:
        barcode_label_info.append(line)
