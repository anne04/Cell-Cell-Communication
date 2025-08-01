import torch
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import random
import numpy as np

def shuffle_data(
    training_set: list()
    ):
    """
    Shuffles the training data
    """
    sample_count = training_set[0].shape[0]
    index_order = np.arange(sample_count)
    random.shuffle(index_order)
    # now reorder training data in that order
    

class fusionMLP(torch.nn.Module):
    def __init__(self, 
                 input_size: np.int = 512 + 264 + 1023, 
                 hidden_size_fusion: np.int = 1024, 
                 output_size_fusion: np.int = 256,
                 hidden_size_predictor_layer1: np.int = 256*2,
                 hidden_size_predictor_layer2: np.int = 256
                ):
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
        concat_fused_emb = torch.cat(fused_emb_sender, fused_emb_rcvr)
        ppi_prediction = self.ppi_predict_layer(concat_fused_emb)
        
        return ppi_prediction

def train_fusionMLP(
    training_set: list(),
    validation_set: list(),
    epoch: int = 1000,
    batch_size: int = 32,
    learning_rate: np.float =  1e-4
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
    # initialize the model
    model_fusionMLP = fusionMLP(
                 input_size = 512 + 264 + 1023, 
                 hidden_size_fusion = 1024, 
                 output_size_fusion = 256,
                 hidden_size_predictor_layer1 = 256*2,
                 hidden_size_predictor_layer2 = 256
    )

    # set the loss function
    loss_fn = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.Adam(model_fusionMLP.parameters(), lr=learing_rate)

    total_training_samples = training_set[0].shape[0]
    total_batch = total_training_samples//batch_size
    min_loss = 10000 # just a big number to initialize
    for epoch_indx in range (0, epoch):
        # shuffle the training set
        shuffle_data(training_set)
        training_sender_emb = training_set[0]
        training_rcv_emb = training_set[1]
        training_prediction = training_set[2]
        
        # model_fusionMLP.train() # training mode
        
        total_loss = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            
            batch_sender_emb = training_sender_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :]
            batch_data_rcv_emb = training_rcv_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :]
            batch_target = training_prediction[batch_idx*batch_size: (batch_idx+1)*batch_size, :]
            
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            
            loss = loss_function(batch_prediction, batch_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            
        
        avg_loss = total_loss/batch_size
        if epoch_index%500 == 0:
            print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))
            
            # run validation
            # CHECK: if you use dropout layer, you might need to set some flag during inference step 
            batch_sender_emb = validation_sender_emb
            batch_data_rcv_emb = validation_rcv_emb
            batch_target = validation_prediction
            
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            validation_loss = loss_function(batch_prediction, batch_target)
            print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))
            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_fusionMLP, "my_model_fusionMLP.pickle")
                # model = torch.load("my_model.pickle")
                torch.save(model_fusionMLP.state_dict(), "my_model_fusionMLP_state_dict.pickle")
                # model = nn.Sequential(...)
                # model.load_state_dict(torch.load("my_model.pickle"))

        
        

                   

      




# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
