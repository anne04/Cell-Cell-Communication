# Written By 
# Fatema Tuz Zohora

import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from torch_geometric.data import DataLoader






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--data_name', type=str, help='Name of the dataset', required=True) #default='PDAC_64630', 
    parser.add_argument( '--model_name', type=str, help='Provide a model name', required=True)
    parser.add_argument( '--run_id', type=int, help='Please provide a running ID, for example: 0, 1, 2, etc. Five runs are recommended.', required=True )
    parser.add_argument( '--model_type', type=str, help='Provide a model type: vgae or dgi', required=True)
    #=========================== default is set ======================================
    parser.add_argument( '--vgae_encoder', type=str, default='gcn', help='Provide an encoder: gcn or gat')
    parser.add_argument( '--num_epoch', type=int, default=60000, help='Number of epochs or iterations for model training')
    parser.add_argument( '--epoch_interval', type=int, default=500, help='Number of epochs or iterations interval.')
    parser.add_argument( '--model_path', type=str, default='model/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to save the node embedding and attention scores') 
    parser.add_argument( '--hidden', type=int, default=512, help='Hidden layer dimension (dimension of node embedding)')
#    parser.add_argument( '--hidden_2', type=int, default=256, help='Hidden layer dimension (dimension of node embedding)')
    parser.add_argument( '--training_data', type=str, default='input_graph/', help='Path to input graph. ')
    parser.add_argument( '--heads', type=int, default=1, help='Number of heads in the attention model')
    parser.add_argument( '--dropout', type=float, default=0)
    parser.add_argument( '--lr_rate', type=float, default=0.0001)
    parser.add_argument( '--manual_seed', type=str, default='no')
    parser.add_argument( '--seed', type=int )
    parser.add_argument( '--tanh', type=int, default=0)
    parser.add_argument( '--multi_graph', type=int, default=0)
    #parser.add_argument( '--split', type=int, default=0)
    parser.add_argument( '--total_subgraphs', type=int, default=1)
    parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
    parser.add_argument( '--BCE_row_count', type=int, default=5000, help='BCE_row_count')
    parser.add_argument( '--BCE_weight_flag', type=int, default=0, help='Weighted BCE or not')
    #=========================== optional ======================================
    parser.add_argument( '--load', type=int, default=0, help='Load a previously saved model state')  
    parser.add_argument( '--load_model_name', type=str, default='None' , help='Provide the model name that you want to reload')
    #============================================================================
    args = parser.parse_args() 

    #parser.add_argument( '--options', type=str)
    #parser.add_argument( '--withFeature', type=str, default='r1') 
    #parser.add_argument( '--workflow_v', type=int, default=1)
    #parser.add_argument( '--datatype', type=str)

    '''
    if args.total_subgraphs > 1 :
        args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'graph_bag'
    else:
        args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'adjacency_records'
    '''
    if args.training_data=="input_graph/":
        args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'adjacency_gene_records' #_1D'

    if args.total_subgraphs > 1 :
        node_id_sorted = args.metadata_to + args.data_name + '/'+ args.data_name+'_'+'gene_node_id_sorted_xy'

    args.embedding_path = args.embedding_path + args.data_name +'/'
    args.model_path = args.model_path + args.data_name +'/'
    args.model_name = args.model_name + '_r' + str(args.run_id)



    print(args.data_name+', '+str(args.heads)+', '+args.training_data+', '+str(args.hidden) )

    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path) 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 

    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ###### adding multiple samples together ###########
    training_data = []
    training_data.append(args.training_data)
    training_data.append("input_graph/"+"LRbind_lymph_1D_manualDB_geneLocalCorrKNN_bidir_negatome/"+"LRbind_lymph_1D_manualDB_geneLocalCorrKNN_bidir_negatome"+ '_' + 'adjacency_gene_records')



    if args.total_subgraphs == 1:
        if args.model_type == 'dgi':
            if args.tanh == 1: 
                from LRbind_model_tanh import get_graph, train_NEST
                if args.multi_graph == 1: 
                    from LRbind_model_tanh import get_multiGraph, train_multigraph_NEST

                print('Using Tanh activation function for attention layer')
            else:
                from LRbind_model import get_graph, train_NEST
            
            if args.multi_graph == 1: 
                # data preparation
                data_loader, num_feature = get_multiGraph(training_data) 
                # train the model
                DGI_model = train_multigraph_NEST(args, data_loader=data_loader, in_channels=int(num_feature), ['LUAD', 'LYMPH'])
            else:
                # data preparation
                data_loader, num_feature = get_graph(args.training_data)    
                # train the model
                DGI_model = train_NEST(args, data_loader=data_loader, in_channels=int(num_feature))
        # training done
        elif args.model_type == 'vgae':
            from LRbind_VGAE_model import get_graph, train_LRbind
            # data preparation
            data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input  = get_graph(args.training_data)    
            # train the model
            VGAEModel_model = train_LRbind(args, data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input)
            # training done
        elif args.model_type == 'vgae-hetero':
            from LRbind_VGAE_model_hetero import get_graph, train_LRbind
            # data preparation
            data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input  = get_graph(args.training_data)    
            # train the model
            VGAEModel_model = train_LRbind(args, data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input)
            # training done
        elif args.model_type == 'dgi-hetero':
            from LRbind_model_heterogenous import get_graph, train_NEST
            # data preparation
            data_loader, num_feature = get_graph(args.training_data)    
            # train the model
            DGI_model = train_NEST(args, data_loader=data_loader, in_channels=int(num_feature))

        else:
            print('error input')
    elif args.total_subgraphs > 1:
        if args.model_type == 'dgi':
            from LRbind_model_split import get_split_graph, train_NEST #_v2
            # data preparation
            # graph_bag, num_feature = get_graph(args.training_data)
            graph_bag, num_feature = get_split_graph(args.training_data, node_id_sorted, args.total_subgraphs)    
            # train the model
            DGI_model = train_NEST(args, graph_bag=graph_bag, in_channels=num_feature)
            # training done

        # training done
        elif args.model_type == 'vgae':
            from LRbind_VGAE_model import get_graph, train_LRbind
            # data preparation
            data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input  = get_graph(args.training_data)    
            # train the model
            VGAEModel_model = train_LRbind(args, data_loader, num_feature, adj_list_dict, num_nodes, total_adjacency_input)
            # training done
        else:
            print('error')
    # you can do something with the model here




    