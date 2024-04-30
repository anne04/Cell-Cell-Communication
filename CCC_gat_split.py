from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax #Linear, 
from torch_geometric.data import Data, DataLoader
import gzip

from GATv2Conv_NEST import GATv2Conv

def get_graph(training_data):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
    Returns:
        List of torch_geometric.data.Data type: Contains the input graph
        Integer: Dimension of node embedding
    """
    
    f = gzip.open(training_data , 'rb')
    graph_bag, num_feature = pickle.load(f)
    
    ###########
    # split it into N set of edges
    ###########

    return graph_bag, num_feature #graph_bags


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """

        
        super(Encoder, self).__init__()
        print('incoming channel %d'%in_channels)

        heads = heads
        self.conv =  GATv2Conv(in_channels, hidden_channels, edge_dim=3, heads=heads, concat = False)#, dropout=dropout)
        self.conv_2 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=3, heads=heads, concat = False)#, dropout=0)

        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'

        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'

        #self.prelu = nn.Tanh(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)


    def forward(self, data):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """

        # layer 1
        x, attention_scores, attention_scores_unnormalized = self.conv(data.x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights = True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized


        # layer 2
        x, attention_scores, attention_scores_unnormalized  = self.conv_2(x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights = True)  # <---- ***
        self.attention_scores_mine = attention_scores #self.attention_scores_mine_l1 #attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized #self.attention_scores_mine_unnormalized_l1 #attention_scores_unnormalized

        ###############################
        x = self.prelu(x)

        return x #, attention_scores

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    #print('inside corruption function')
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)


def train_NEST(args, graph_bag, in_channels):
    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    loss_curve = np.zeros((args.num_epoch//500+1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden, heads=args.heads, dropout = args.dropout),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate) #1e-5)#5 #6
    DGI_filename = args.model_path+'DGI_'+ args.model_name  +'.pth.tar'

    if args.load:
        DGI_load_path = args.model_path+'DGI_'+ args.load_model_name+'.pth.tar'
        DGI_model.load_state_dict(torch.load(DGI_load_path))
        DGI_optimizer.load_state_dict(torch.load(args.model_path+'DGI_optimizer_'+ args.load_model_name  +'.pth.tar'))

    import datetime
    start_time = datetime.datetime.now()
    min_loss=10000
    print('Saving init model state ...')
    torch.save(DGI_model.state_dict(), args.model_path+'DGI_init'+ args.model_name  + '.pth.tar')
    torch.save(DGI_optimizer.state_dict(), args.model_path+'DGI_optimizer_init'+ args.model_name  + '.pth.tar')
    #print('training starts ...')
    for epoch in range(args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for subgraph in range (0, len(graph_bag)):
            
            X_data = graph_bag[subgraph][0]
            row_col_temp = graph_bag[subgraph][1]
            edge_weight_temp = graph_bag[subgraph][2]
            
            X = torch.tensor(X_data, dtype=torch.float)
            edge_index = torch.tensor(np.array(row_col_temp), dtype=torch.long).T
            edge_attr = torch.tensor(np.array(edge_weight_temp), dtype=torch.float)
            
            data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
            data_loader = DataLoader([data], batch_size=1)
            for data in data_loader:
                data = data.to(device)
                pos_z, neg_z, summary = DGI_model(data=data)
                DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
                DGI_loss.backward()
                DGI_all_loss.append(DGI_loss.item())
                
        DGI_optimizer.step()

        if ((epoch)%500) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(DGI_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            if np.mean(DGI_all_loss)<min_loss:

                min_loss=np.mean(DGI_all_loss)

                # save the current model state
                torch.save(DGI_model.state_dict(), DGI_filename)
                torch.save(DGI_optimizer.state_dict(), args.model_path+'DGI_optimizer_'+ args.model_name  +'.pth.tar')
                save_tupple=[pos_z, neg_z, summary] 
                ############################################################################################################
                for subgraph in range (0, len(graph_bag)):
                
                    X_data = graph_bag[subgraph][0]
                    row_col_temp = graph_bag[subgraph][1]
                    edge_weight_temp = graph_bag[subgraph][2]
                    
                    X = torch.tensor(X_data, dtype=torch.float)
                    edge_index = torch.tensor(np.array(row_col_temp), dtype=torch.long).T
                    edge_attr = torch.tensor(np.array(edge_weight_temp), dtype=torch.float)
                    
                    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
                    data_loader = DataLoader([data], batch_size=1)
                    for data in data_loader:
                        data = data.to(device)
                        pos_z, neg_z, summary = DGI_model(data=data)              
               
                    # save the node embedding
                    X_embedding = pos_z
                    X_embedding = X_embedding.cpu().detach().numpy()
                    X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
                    with gzip.open(X_embedding_filename+'_subgraph'+str(subgraph), 'wb') as fp:  
                        pickle.dump(X_embedding, fp)
                        

                    # save the attention scores
    
                    X_attention_index = DGI_model.encoder.attention_scores_mine[0]
                    X_attention_index = X_attention_index.cpu().detach().numpy()
    
                    # layer 1
                    X_attention_score_normalized_l1 = DGI_model.encoder.attention_scores_mine_l1[1]
                    X_attention_score_normalized_l1 = X_attention_score_normalized_l1.cpu().detach().numpy()
                    # layer 1 unnormalized
                    X_attention_score_unnormalized_l1 = DGI_model.encoder.attention_scores_mine_unnormalized_l1
                    X_attention_score_unnormalized_l1 = X_attention_score_unnormalized_l1.cpu().detach().numpy()
    
                    # layer 2
                    X_attention_score_normalized = DGI_model.encoder.attention_scores_mine[1]
                    X_attention_score_normalized = X_attention_score_normalized.cpu().detach().numpy()
                    # layer 2 unnormalized
                    X_attention_score_unnormalized = DGI_model.encoder.attention_scores_mine_unnormalized
                    X_attention_score_unnormalized = X_attention_score_unnormalized.cpu().detach().numpy()
    
                    print('making the bundle to save')
                    X_attention_bundle = [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized, X_attention_score_unnormalized_l1, X_attention_score_normalized]
                    X_attention_filename =  args.embedding_path + args.model_name + '_attention'+'_subgraph'+str(subgraph)
                    with gzip.open(X_attention_filename, 'wb') as fp:  
                        pickle.dump(X_attention_bundle, fp)
                ############################################################################################################################
                logfile=open(args.model_path+'DGI_'+ args.model_name+'_loss_curve.csv', 'wb')
                np.savetxt(logfile,loss_curve, delimiter=',')
                logfile.close()

                #print(DGI_model.encoder.attention_scores_mine_unnormalized_l1[0:10])

#            if ((epoch)%60000) == 0:
#                DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)  #5 #6

    end_time = datetime.datetime.now()

#        torch.save(DGI_model.state_dict(), DGI_filename)
    print('Training time in seconds: ', (end_time-start_time).seconds)
    DGI_model.load_state_dict(torch.load(DGI_filename))
    print("debug loss")
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print("debug loss latest tupple %g"%DGI_loss.item())
    DGI_loss = DGI_model.loss(save_tupple[0], save_tupple[1], save_tupple[2])
    print("debug loss min loss tupple %g"%DGI_loss.item())

    return DGI_model

