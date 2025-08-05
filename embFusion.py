import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import random
import numpy as np


cellNEST_dimension = 512
lrbind_dimension = 264
proteinEmb_dimension = 1024


def split_branch(data):
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024
    sender_emb = data[:, 0:sender_dimension_total]    
    rcv_emb = data[:, sender_dimension_total:sender_dimension_total+rcvr_dimension_total]
    prediction = data[:, prediction_column]
    return sender_emb, rcv_emb, prediction 
        
def shuffle_data(
    training_set #: torch.tensor 
    ):
    """
    Shuffles the training data
    """
    # Generate random permutation of row indices
    sample_count = training_set.size(0)
    prediction_column = training_set.size(1)-1
    row_perm = torch.randperm(sample_count)

    # Shuffle the rows using advanced indexing
    training_set = training_set[row_perm]
    print(training_set)
    training_sender_emb, training_rcv_emb, training_prediction = split_branch(training_set)
    """
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024
        
    training_sender_emb = training_set[:, 0:sender_dimension_total]    
    training_rcv_emb = training_set[:, sender_dimension_total:sender_dimension_total+rcvr_dimension_total]
    training_prediction = training_set[:, prediction_column]
    """
    return training_sender_emb, training_rcv_emb, training_prediction 
    
    
def data_to_tensor(
    training_set #: list()
    ):
    """
    training_set = list of [sender_emb, rcvr_emb, pred]
    """
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024
    training_set_matrix = np.zeros((len(training_set), sender_dimension_total + rcvr_dimension_total + 1 )) # 1=prediction column
    for i in range(0, len(training_set)):
        training_set_matrix[i, 0:sender_dimension_total] = np.concatenate((training_set[i][0][0],training_set[i][0][1], training_set[i][0][2]), axis=0)
        
        training_set_matrix[i, sender_dimension_total:sender_dimension_total+rcvr_dimension_total] = np.concatenate((training_set[i][1][0],training_set[i][1][1], training_set[i][1][2]), axis=0)
        
        training_set_matrix[i, sender_dimension_total+rcvr_dimension_total] = training_set[i][2]

    # convert to tensor
    training_set = torch.tensor(training_set_matrix, dtype=torch.float)
    return training_set
    
class fusionMLP(torch.nn.Module):
    def __init__(self, 
                 input_size: np.int32 = 512 + 264 + 1023, 
                 hidden_size_fusion: np.int32 = 1024, 
                 output_size_fusion: np.int32 = 256,
                 hidden_size_predictor_layer1: np.int32 = 256*2,
                 hidden_size_predictor_layer2: np.int32 = 256
                ):
        super().__init__() # without this error happens
        # Branch: sender
        self.sender_fusion_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size_fusion),
          nn.ReLU(),
          nn.Linear(hidden_size_fusion, output_size_fusion),
          nn.ReLU()
        )
                     
        # Branch: rcvr
        self.rcvr_fusion_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size_fusion),
          nn.ReLU(),
          nn.Linear(hidden_size_fusion, output_size_fusion),
          nn.ReLU()
        )

        # concat both and pass through another MLP
        self.ppi_predict_layer = nn.Sequential(
            nn.Linear(output_size_fusion*2, hidden_size_predictor_layer1),
            nn.ReLU(),
            nn.Linear(hidden_size_predictor_layer1, hidden_size_predictor_layer2),
            nn.ReLU(),
            nn.Linear(hidden_size_predictor_layer2, 1),
            nn.Sigmoid()
        )

    def forward(self, sender_emb, receiver_emb):
        fused_emb_sender = self.sender_fusion_layer(sender_emb)
        fused_emb_rcvr = self.rcvr_fusion_layer(sender_emb)
        concat_fused_emb = torch.cat((fused_emb_sender, fused_emb_rcvr), dim=1)
        ppi_prediction = self.ppi_predict_layer(concat_fused_emb)
        
        return ppi_prediction

def train_fusionMLP(
    training_set#: torch.tensor,
    validation_set#: torch.tensor = None,
    epoch: int = 1000,
    batch_size: int = 32,
    learning_rate: float =  1e-4,
    ):
    """
    split the training set into 80% training data and 20% validation set
    """
    """
    # CHECK = do I need to set a seed?
    random.shuffle(dataset)
    training_set_count = (len(dataset)*80)//100
    training_set = dataset[0:training_set_count]
    validation_set = dataset[training_set_count:]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    model_fusionMLP = fusionMLP(
                 input_size = 512 + 264 + 1024, 
                 hidden_size_fusion = 1024, 
                 output_size_fusion = 256,
                 hidden_size_predictor_layer1 = 256*2,
                 hidden_size_predictor_layer2 = 256
    ).to(device)

    # set the loss function
    loss_fn = nn.MSELoss() #CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.Adam(model_fusionMLP.parameters(), lr=learning_rate)
    total_training_samples = training_set.shape[0]
    total_batch = total_training_samples//batch_size

    loss_curve = np.zeros((args.num_epoch//args.epoch_interval+1))
    loss_curve_counter = 0




    min_loss = 10000 # just a big number to initialize
    for epoch_indx in range (0, epoch):
        # shuffle the training set
        training_sender_emb, training_rcv_emb, training_prediction = shuffle_data(training_set)        
        # model_fusionMLP.train() # training mode
        total_loss = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            # .to(device) to transfer to GPU
            batch_sender_emb = training_sender_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_data_rcv_emb = training_rcv_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_target = training_prediction[batch_idx*batch_size: (batch_idx+1)*batch_size].to(device)

            # move the sender and rcvr emb to the GPU
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            
            loss = loss_fn(batch_prediction.flatten(), batch_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            
        
        avg_loss = total_loss/total_batch
        if epoch_indx%args.epoch_interval == 0:
            print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))
            
            # run validation
            # CHECK: if you use dropout layer, you might need to set some flag during inference step 
            validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(training_set)
            # .to(device) to transfer to GPU
            batch_sender_emb = validation_sender_emb.to(device)
            batch_data_rcv_emb = validation_rcv_emb.to(device)
            batch_target = validation_prediction.to(device)
            
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            validation_loss = loss_function(batch_prediction.flatten(), batch_target)
            print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))
            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_fusionMLP, "model/my_model_fusionMLP.pickle")
                # model = torch.load("my_model.pickle")
                torch.save(model_fusionMLP.state_dict(), "model/my_model_fusionMLP_state_dict.pickle")
                # model = nn.Sequential(...)
                # model.load_state_dict(torch.load("my_model.pickle"))
                print('*** min loss found! ***')
    
                
        

                   

      




# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
