 
import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pathologist_label_file='/cluster/home/t116508uhn/64630/tumor_64630_D1_IX_annotation.csv' # IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_label=[]
for i in range (1, len(pathologist_label)):
  if pathologist_label[i][1] == 'tumor': #'Tumour':
      barcode_label.append([pathologist_label[i][0]])
      
'''barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=dict()
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info[line[0]] = 1
        i=i+1
        
for i in range (0, len(barcode_label)):   
    if barcode_label[i][0] not in barcode_info:
        print('not found!')'''
    
barcode_filename = '/cluster/home/t116508uhn/64630/tumor_whitelist.csv'
f=open(barcode_filename, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
# no header, just line seperated list
writer.writerows(barcode_label)
f.close()
      
