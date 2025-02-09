# Written By 
# Fatema Tuz Zohora

from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax #Linear, 
from torch_geometric.data import Data, DataLoader
import gzip

from LRbind_VGAEModel import VGAEModel

def get_graph(training_data):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
    Returns:
        List of torch_geometric.data.Data type: Loaded input graph
        Integer: Dimension of node embedding
    """
    
    f = gzip.open(training_data , 'rb') # read input graph
    row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node = pickle.load(f)

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
    for i in range (0, len(gene_node_type)):
        X[i][:] = feature_vector[gene_node_type[i]]
    
    X_data = X # node feature vector
    print('Node feature matrix: X has dimension ', X_data.shape)
    print("Total number of edges in the input graph is %d"%len(row_col_gene))
    
    

    ###########
    for i in range (0, len(edge_weight)):
        edge_weight[i]=edge_weight[i][0] # making it 1D list
        
    edge_index = torch.tensor(np.array(row_col_gene), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)

    print('Input graph generation done')

    data_loader = DataLoader(graph_bags, batch_size=1) 
    
    return data_loader, num_feature



class my_data():
    def __init__(self, x, edge_index, edge_attr):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def train_NEST(args, data_loader, in_channels):
    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    loss_curve = np.zeros((args.num_epoch//500+1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VGAEModel_model = VGAEModel(in_channels, args.hidden, args.hidden).to(device)
    VGAEModel_optimizer = torch.optim.Adam(VGAEModel_model.parameters(), lr=args.lr_rate) 
    VGAEModel_filename = args.model_path+'VGAEModel_'+ args.model_name  +'.pth.tar'

    if args.load == 1:
        print('loading model')
        checkpoint = torch.load(VGAEModel_filename)
        VGAEModel_model.load_state_dict(checkpoint['model_state_dict'])
        VGAEModel_model.to(device)
        VGAEModel_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        '''
        for state in VGAEModel_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        ''' 
        print('min_loss was %g'%min_loss)
    else:
        print('Saving init model state ...')
        torch.save({
            'epoch': 0,
            'model_state_dict': VGAEModel_model.state_dict(),
            'optimizer_state_dict': VGAEModel_optimizer.state_dict(),
            #'loss': loss,
            }, args.model_path+'VGAEModel_init_model_optimizer_'+ args.model_name  + '.pth.tar')
        min_loss = 10000
        epoch_start = 0
        
    import datetime
    start_time = datetime.datetime.now()

    #print('training starts ...')
    for epoch in range(epoch_start, args.num_epoch):
        VGAEModel_model.train()
        VGAEModel_optimizer.zero_grad()
        VGAEModel_all_loss = []

        for data in data_loader:
            data = data.to(device)
            logits, X_embedding =  = VGAEModel_model(data=data) # output from decoder -> adj
            # compute loss
            VGAEModel_loss = norm * F.binary_cross_entropy(
                logits.view(-1), adj.view(-1), weight=weight_tensor
            )
            kl_divergence = (
                0.5
                / logits.size(0)
                * (
                    1
                    + 2 * vgae_model.log_std
                    - vgae_model.mean**2
                    - torch.exp(vgae_model.log_std) ** 2
                )
                .sum(1)
                .mean()
            )
            VGAEModel_loss -= kl_divergence
    
            # backward
            VGAEModel_optimizer.zero_grad()
            VGAEModel_loss.backward()
            VGAEModel_all_loss.append(VGAEModel_loss.item())
            VGAEModel_optimizer.step()
            

        if ((epoch)%500) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(VGAEModel_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(VGAEModel_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            if np.mean(VGAEModel_all_loss)<min_loss:

                min_loss=np.mean(VGAEModel_all_loss)

                ######## save the current model state ########
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': VGAEModel_model.state_dict(),
                    'optimizer_state_dict': VGAEModel_optimizer.state_dict(),
                    'loss': min_loss,
                    }, VGAEModel_filename)

                ##################################################
                # save the node embedding
                X_embedding = X_embedding.cpu().detach().numpy()
                X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
                with gzip.open(X_embedding_filename, 'wb') as fp:  
                    pickle.dump(X_embedding, fp)
                                    
  
                logfile=open(args.model_path+'VGAEModel_'+ args.model_name+'_loss_curve.csv', 'wb')
                np.savetxt(logfile,loss_curve, delimiter=',')
                logfile.close()


    end_time = datetime.datetime.now()

    print('Training time in seconds: ', (end_time-start_time).seconds)

    checkpoint = torch.load(VGAEModel_filename)
    VGAEModel_model.load_state_dict(checkpoint['model_state_dict'])
    VGAEModel_model.to(device)
    VGAEModel_model.eval()
    print("debug loss")
    VGAEModel_loss = VGAEModel_model.loss(data)
    print("debug loss latest tupple %g"%VGAEModel_loss.item())

    return VGAEModel_model

