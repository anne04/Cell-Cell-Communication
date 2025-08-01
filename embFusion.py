import torch
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import random
import numpy as np


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

    
    for epoch_indx in range (0, epoch):
        # shuffle the training set
        shuffle_data(training_set)
        training_sender_emb = training_set
        # model_fusionMLP.train() # training mode
        optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
        total_loss = [] 
        for batch_idx in range(0, total_batch):
            
            
            batch_data = training_set[batch_idx*batch_size: (batch_idx+1)*batch_size]
            batch_data
            
        sender_emb_batch = training_set
        prediction = model_fusionMLP(inputs)

        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()


    
        
        

                   

      




# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
