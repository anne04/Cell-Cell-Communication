# Written By 
# Fatema Tuz Zohora

from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax #, GATv2Conv #Linear, 
from torch_geometric.data import Data, DataLoader
import gzip

from GATv2Conv_NEST import GATv2Conv

def get_multiGraph(training_data_list):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
    Returns:
        List of torch_geometric.data.Data type: Loaded input graph
        Integer: Dimension of node embedding
    """
    num_feature_list = []
    for training_data in training_data_list:
        f = gzip.open(training_data , 'rb') # read input graph
        row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(f)
        num_feature_list.append(np.max(np.unique(gene_node_type))+1)
        print('Unique gene type: %d'%np.max(np.unique(gene_node_type)))


    num_feature = np.max(num_feature_list) 
    graph_bags = []
    for training_data in training_data_list:
        f = gzip.open(training_data , 'rb') # read input graph
        row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(f)
        # one hot vector used as node feature vector
        feature_vector = np.eye(num_feature, num_feature)
        # 1 0 0 0 = feature_vector[0]
        # 0 1 0 0 = feature_vector[1]
        # 0 0 1 0 = feature_vector[2]
        # 0 0 0 1 = feature_vector[3]
        # feature_vector[feature_type]
        print('total_num_gene_node %d, len gene_node_type %d'%(total_num_gene_node, len(gene_node_type)))
        X = np.zeros((total_num_gene_node, num_feature))
        print(len(gene_node_expression))
        for i in range (0, len(gene_node_type)):
            #print(i)
            X[i][:] = feature_vector[gene_node_type[i]]*gene_node_expression[i]
        
        X_data = X # node feature vector
        print('Node feature matrix: X has dimension ', X_data.shape)
        print("Total number of edges in the input graph is %d"%len(row_col_gene))
        

        ###########

        edge_index = torch.tensor(np.array(row_col_gene), dtype=torch.long).T
        edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

        
        graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
        graph_bags.append(graph)

    print('Input graph generation done')

    data_loader = DataLoader(graph_bags, batch_size=1) 
    
    return data_loader, num_feature



def get_graph(training_data):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
    Returns:
        List of torch_geometric.data.Data type: Loaded input graph
        Integer: Dimension of node embedding
    """
    
    f = gzip.open(training_data , 'rb') # read input graph
    row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(f)
    
    print('Unique gene type: %d'%np.max(np.unique(gene_node_type)))
    num_feature = np.max(np.unique(gene_node_type))+1
    
    # one hot vector used as node feature vector
    feature_vector = np.eye(num_feature, num_feature)
    # 1 0 0 0 = feature_vector[0]
    # 0 1 0 0 = feature_vector[1]
    # 0 0 1 0 = feature_vector[2]
    # 0 0 0 1 = feature_vector[3]
    # feature_vector[feature_type]
    print('total_num_gene_node %d, len gene_node_type %d'%(total_num_gene_node, len(gene_node_type)))
    X = np.zeros((total_num_gene_node, num_feature))
    print(len(gene_node_expression))
    for i in range (0, len(gene_node_type)):
        #print(i)
        X[i][:] = feature_vector[gene_node_type[i]]*gene_node_expression[i]
    
    X_data = X # node feature vector
    print('Node feature matrix: X has dimension ', X_data.shape)
    print("Total number of edges in the input graph is %d"%len(row_col_gene))
    

    ###########

    edge_index = torch.tensor(np.array(row_col_gene), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)

    print('Input graph generation done')

    data_loader = DataLoader(graph_bags, batch_size=1) 
    
    return data_loader, num_feature


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        
        super(Encoder, self).__init__()
        print('incoming channel %d'%in_channels)

        heads = heads
        self.conv =  GATv2Conv(in_channels, hidden_channels, edge_dim=1, heads=heads, concat = False, dropout=dropout)
#        self.conv_2 =  GATv2Conv(hidden_channels, 264, edge_dim=1, heads=heads, concat = False)#, dropout=0)
#        self.conv_3 =  GATv2Conv(264, 264, edge_dim=1, heads=heads, concat = False)#, dropout=0)
        self.conv_2 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=1, heads=heads, concat = False, dropout=dropout)
        self.conv_3 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=1, heads=heads, concat = False, dropout=dropout)

        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'

        self.attention_scores_mine_l2 = 'attention_l2'
        self.attention_scores_mine_unnormalized_l2 = 'attention_unnormalized_l2'

        self.attention_scores_mine_l3 = 'attention_l3'
        self.attention_scores_mine_unnormalized_l3 = 'attention_unnormalized_l3'



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
        #self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l2 = attention_scores_unnormalized

        
        
        # layer 3
        #x= self.conv_3(x, data.edge_index, edge_attr=data.edge_attr)  # <---- ***
        x, attention_scores, attention_scores_unnormalized  = self.conv_3(x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights = True)  # <---- ***
        #self.attention_scores_mine = attention_scores #self.attention_scores_mine_l1 #attention_scores
        self.attention_scores_mine_unnormalized_l3 = attention_scores_unnormalized #self.attention_scores_mine_unnormalized_l1 #attention_scores_unnormalized

        

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


def train_NEST(args, data_loader, in_channels):
    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    loss_curve = np.zeros((args.num_epoch//args.epoch_interval+1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden, heads=args.heads, dropout = args.dropout),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    
    #print('initialized DGI model')
    #DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=0.005, weight_decay=5e-4)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate) #1e-5)#5 #6 #DGI_optimizer = torch.optim.RMSprop(DGI_model.parameters(), lr=1e-5)
    DGI_filename = args.model_path+'DGI_'+ args.model_name  +'.pth.tar'

    if args.load == 1:
        print('loading model')
        checkpoint = torch.load(DGI_filename)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        '''
        for state in DGI_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        ''' 
        print('min_loss was %g'%min_loss)
    else:
        print('Saving init model state ...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
            #'loss': loss,
            }, args.model_path+'DGI_init_model_optimizer_'+ args.model_name  + '.pth.tar')
        min_loss = 10000
        epoch_start = 0
        
    import datetime
    start_time = datetime.datetime.now()

    #print('training starts ...')
    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()
            DGI_all_loss.append(DGI_loss.item())
            DGI_optimizer.step()

        if ((epoch)%args.epoch_interval) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(DGI_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            if np.mean(DGI_all_loss)<min_loss:

                min_loss=np.mean(DGI_all_loss)

                ######## save the current model state ########
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DGI_model.state_dict(),
                    'optimizer_state_dict': DGI_optimizer.state_dict(),
                    'loss': min_loss,
                    }, DGI_filename)

                ##################################################
                print("Lower loss found! Save the node embedding")
               
                X_embedding = pos_z
                X_embedding = X_embedding.cpu().detach().numpy()
                X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
                with gzip.open(X_embedding_filename, 'wb') as fp:  
                    pickle.dump(X_embedding, fp)
                                    
                # save the attention scores
                
                X_attention_index = DGI_model.encoder.attention_scores_mine_l1[0]
                X_attention_index = X_attention_index.cpu().detach().numpy()
                '''
                # layer 1
                X_attention_score_normalized_l1 = DGI_model.encoder.attention_scores_mine_l1[1]
                X_attention_score_normalized_l1 = X_attention_score_normalized_l1.cpu().detach().numpy()
                '''
                # layer 1 unnormalized
                X_attention_score_unnormalized_l1 = DGI_model.encoder.attention_scores_mine_unnormalized_l1
                X_attention_score_unnormalized_l1 = X_attention_score_unnormalized_l1.cpu().detach().numpy()

                # layer 2 unnormalized
                X_attention_score_unnormalized_l2 = DGI_model.encoder.attention_scores_mine_unnormalized_l2
                X_attention_score_unnormalized_l2 = X_attention_score_unnormalized_l2.cpu().detach().numpy()
                
                # layer 3
                #X_attention_score_normalized = DGI_model.encoder.attention_scores_mine[1]
                #X_attention_score_normalized = X_attention_score_normalized.cpu().detach().numpy()
                # layer 3 unnormalized
                X_attention_score_unnormalized_l3 = DGI_model.encoder.attention_scores_mine_unnormalized_l3
                X_attention_score_unnormalized_l3 = X_attention_score_unnormalized_l3.cpu().detach().numpy()

                print('making the bundle to save')
                X_attention_bundle = [X_attention_index, X_attention_score_unnormalized_l1, X_attention_score_unnormalized_l2, X_attention_score_unnormalized_l3]
                X_attention_filename =  args.embedding_path + args.model_name + '_attention' #.npy
                # np.save(X_attention_filename, X_attention_bundle) # this is deprecated
                with gzip.open(X_attention_filename, 'wb') as fp:  
                    pickle.dump(X_attention_bundle, fp)
                




                logfile=open(args.model_path+'DGI_'+ args.model_name+'_loss_curve.csv', 'wb')
                np.savetxt(logfile,loss_curve, delimiter=',')
                logfile.close()

                #print(DGI_model.encoder.attention_scores_mine_unnormalized_l1[0:10])

#            if ((epoch)%60000) == 0:
#                DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)  #5 #6

    end_time = datetime.datetime.now()

    print('Training time in seconds: ', (end_time-start_time).seconds)

    checkpoint = torch.load(DGI_filename)
    DGI_model.load_state_dict(checkpoint['model_state_dict'])
    DGI_model.to(device)
    DGI_model.eval()
    print("debug loss")
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print("debug loss latest tupple %g"%DGI_loss.item())

    return DGI_model
