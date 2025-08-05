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
    
    with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir'+'_dataset_embFusion.pkl', 'rb') as fp:  
    	dataset = pickle.load(fp)

    # shuffle and keep 80% for training, 20% for validation
    np.random.shuffle(dataset)
    training_set = dataset[0:(len(dataset)*80)//100]
    val_set = dataset[(len(dataset)*80)//100:]
    
    training_set, add_set = data_to_tensor(training_set, remove_set)
    # append add_set to the validation set
    for item in add_set:
        val_set.append(item) 
    val_set = data_to_tensor(val_set, None)

    train_fusionMLP(training_set, val_set, epoch = 1000,
                    batch_size = 128, learning_rate= 0.001
    )


    
