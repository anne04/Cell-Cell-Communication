import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from embFusion import data_to_tensor
from embFusion import split_branch
import pickle
import gzip
import pandas as pd

def val_fusionMLP_multiBatch(dataset, model_name, threshold_score = 0.7, total_batch = 1):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    """
    model_fusionMLP = fusionMLP(
                 input_size = 264 + 1024, 
                 hidden_size_fusion = 1024, 
                 output_size_fusion = 256,
                 hidden_size_predictor_layer1 = 256*2,
                 hidden_size_predictor_layer2 = 256
    ).to(device)
    model_fusionMLP.load_state_dict(torch.load(model_name))
    model_fusionMLP.to(device)
    """
    model_fusionMLP = torch.load(model_name)
    model_fusionMLP.to(device)
    batch_size = len(dataset)//total_batch
    
    batch_prediction_combined = []
    for batch_idx in range(0, total_batch):
        print(batch_idx)
        # .to(device) to transfer to GPU
        val_set, na = data_to_tensor(dataset[batch_idx*batch_size: (batch_idx+1)*batch_size], None)
        validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(val_set)
        batch_sender_emb = validation_sender_emb.to(device)
        batch_data_rcv_emb = validation_rcv_emb.to(device)
        # batch_target = validation_prediction.to(device)

        # move the sender and rcvr emb to the GPU
        batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
        batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
        for score in batch_prediction:
            batch_prediction_combined.append(score)

    if (batch_idx+1)*batch_size < len(dataset)-1:
        val_set, na = data_to_tensor(dataset[(batch_idx+1)*batch_size:], None)
        validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(val_set)
        batch_sender_emb = validation_sender_emb.to(device)
        batch_data_rcv_emb = validation_rcv_emb.to(device)
        # batch_target = validation_prediction.to(device)

        # move the sender and rcvr emb to the GPU
        batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
        batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
        for score in batch_prediction:
            batch_prediction_combined.append(score)
        

    prediction_score = batch_prediction_combined
    pred_class = []
    for i in range(0, len(prediction_score)):
        if prediction_score[i] >= threshold_score:
            pred_class.append(1)
        else:
            pred_class.append(0)

    
    return prediction_score, pred_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--model_name', type=str, default="embFusion_gene_prot", help='Provide a model name')
    parser.add_argument( '--model_path', type=str, default='embFusion_model', help='Path to save the model state') # We do not need this for output generation 
    parser.add_argument( '--lr_lrbind_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir/model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L_allLR_nodeInfo.csv', 
                        help='Name of the dataset') #, required=True)
    #=========================== default is set ======================================
    parser.add_argument( '--data_to_predict', type=str, \
                    default='/cluster/projects/schwartzgroup/fatema/LRbind/database/LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_dataset_results_to_embFusion.pkl', \
                    help='Path to input graph. ')
    #============================================================================
    args = parser.parse_args() 


    args.model_path = args.model_path +'/'
    args.model_name = args.model_path + args.model_name 

    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ccc_pairs = pd.read_csv(args.lr_lrbind_csv_path, sep=",")
    with gzip.open(args.data_to_predict, 'rb') as fp:  
    	dataset, record_index = pickle.load(fp)

    
   

    model_name = args.model_name
    
    #val_set, na = data_to_tensor(dataset, None)
    prediction_score, pred_class = val_fusionMLP_multiBatch(dataset, model_name, threshold_score=0.7, total_batch = 1000)    

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
    ccc_pairs['pred_score'] = pred_score
    # save it
    ccc_pairs.to_csv('model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L_allLR_nodeInfo.csv', index=False) 



                
