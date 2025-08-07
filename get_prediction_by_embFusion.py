import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from embFusion import data_to_tensor
from embFusion import train_fusionMLP
from embFusion import val_fusionMLP
import pickle
import gzip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    #parser.add_argument( '--data_name', type=str, help='Name of the dataset') #default='PDAC_64630', 
    parser.add_argument( '--model_name', type=str, default="embFusion_test", help='Provide a model name')
    #=========================== default is set ======================================
    parser.add_argument( '--num_epoch', type=int, default=10000, help='Number of epochs or iterations for model training')
    parser.add_argument( '--model_path', type=str, default='model/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--training_data', type=str, default='database/LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_dataset_embFusion.pkl', help='Path to input graph. ')
    parser.add_argument( '--dropout', type=float, default=0)
    parser.add_argument( '--lr_rate', type=float, default=0.00001)
    parser.add_argument( '--manual_seed', type=str, default='no')
    parser.add_argument( '--seed', type=int )
    #=========================== optional ======================================
    parser.add_argument( '--load', type=int, default=0, help='Load a previously saved model state')  
    parser.add_argument( '--load_model_name', type=str, default='None' , help='Provide the model name that you want to reload')
    #============================================================================
    args = parser.parse_args() 

    args.training_data = args.training_data #+ args.data_name + '/' + args.data_name + '_' + 'adjacency_records'

    
    args.model_path = args.model_path +'/'
    args.model_name = args.model_name 


    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 

    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ccc_pairs = pd.read_csv(args.lr_lrbind_csv_path, sep=",")
    with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir'+'_dataset_results_to_embFusion.pkl', 'rb') as fp:  
    	dataset, record_index = pickle.load(fp)

    
   

    model_name = 'model/my_model_fusionMLP.pickle'
    val_set, na = data_to_tensor(dataset, None)
    prediction_score, pred_class = val_fusionMLP(val_set, model_name, threshold_score=0.7)    

    index_vs_score = dict()
    for i in range(0, len(record_index)):
        index = record_index[i]
        index_vs_score[index] = prediction_score[i]
    
    for i in range(0, len(ccc_pairs)):
        if i not in index_vs_score:
            index_vs_score[i] = -1
    
    pred_score = []
    for i in range(0, len(ccc_pairs)):
        pred_score.append(index_vs_score[i])

    # now add this column to ccc_pairs

    # save it
