import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import torch.nn.functional as F


cellNEST_dimension = 512
lrbind_dimension = 256 #264
proteinEmb_dimension = 1024

    
class fusionMLP(torch.nn.Module):
    def __init__(self, 
                 input_size: np.int32 = lrbind_dimension + 1024,  
                 hidden_size_fusion: np.int32 = 512, #1024, 
                 output_size_fusion: np.int32 = 128, #512, # 256
                 hidden_size_predictor_layer1: np.int32 = 128*2, #512*2,
                 hidden_size_predictor_layer2: np.int32 = 128 #256
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
        fused_emb_rcvr = self.rcvr_fusion_layer(receiver_emb)

        fused_emb_sender = F.normalize(fused_emb_sender, p=2) #, dim=1)
        fused_emb_rcvr = F.normalize(fused_emb_rcvr, p=2) #, dim=1)

        concat_fused_emb = torch.cat((fused_emb_sender, fused_emb_rcvr), dim=1)
        ppi_prediction = self.ppi_predict_layer(concat_fused_emb)

       	return ppi_prediction

def split_branch(data):
    prediction_column = data.size(1)-1
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024 # 512+
    sender_emb = data[:, cellNEST_dimension:sender_dimension_total]    
    rcv_emb = data[:, sender_dimension_total+cellNEST_dimension:sender_dimension_total+rcvr_dimension_total]
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
    #prediction_column = training_set.size(1)-1
    row_perm = torch.randperm(sample_count)

    # Shuffle the rows using advanced indexing
    training_set = training_set[row_perm]
    #print(training_set)
    training_sender_emb, training_rcv_emb, training_prediction = split_branch(training_set)
    """
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024

    training_sender_emb = training_set[:, 0:sender_dimension_total]    
    training_rcv_emb = training_set[:, sender_dimension_total:sender_dimension_total+rcvr_dimension_total]
    training_prediction = training_set[:, prediction_column]
    """
    return training_sender_emb, training_rcv_emb, training_prediction 
    

def data_to_tensor(
    training_set, #: list()
    remove_set=None
    ):
    """
    training_set = list of [sender_emb, rcvr_emb, pred]
    """
    add_set = []
    rcvr_dimension_total = sender_dimension_total = 512 + 264 + 1024
    training_set_matrix = np.zeros((len(training_set), sender_dimension_total + rcvr_dimension_total + 1 )) # 1=prediction column
    for i in range(0, len(training_set)):
        if remove_set != None:
            if training_set[i][3]+'_to_'+training_set[i][4] in remove_set:
                add_set.append(training_set[i])
                continue

        training_set_matrix[i, 0:sender_dimension_total] = np.concatenate((training_set[i][0][0],training_set[i][0][1], training_set[i][0][2]), axis=0)

        training_set_matrix[i, sender_dimension_total:sender_dimension_total+rcvr_dimension_total] = np.concatenate((training_set[i][1][0],training_set[i][1][1], training_set[i][1][2]), axis=0)

        training_set_matrix[i, sender_dimension_total+rcvr_dimension_total] = training_set[i][2]

    # convert to tensor
    training_set = torch.tensor(training_set_matrix, dtype=torch.float)

    return training_set, add_set




def train_fusionMLP(
    args,
    training_set, #: torch.tensor,
    validation_set, #: torch.tensor = None,
    epoch: np.int32 = 1000,
    batch_size: np.int32 = 32,
    learning_rate: float =  1e-4,
    val_class: list() = None,
    threshold_score = 0.7
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
                 input_size =  lrbind_dimension + 1024, 
                 hidden_size_fusion = 1024, 
                 output_size_fusion = 256,
                 hidden_size_predictor_layer1 = 256*2,
                 hidden_size_predictor_layer2 = 256
    ).to(device)
    print(model_fusionMLP)
    # set the loss function
    loss_function = nn.MSELoss() #CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.Adam(model_fusionMLP.parameters(), lr=learning_rate)
    epoch_interval = 20
    loss_curve = np.zeros((epoch//epoch_interval+1, 2))
    loss_curve_counter = 0
    total_training_samples = training_set.shape[0]
    total_batch = total_training_samples//batch_size
    min_loss = 10000 # just a big number to initialize
    for epoch_indx in range (0, epoch):
        # shuffle the training set
        training_sender_emb, training_rcv_emb, training_prediction = shuffle_data(training_set)        
        # model_fusionMLP.train() # training mode
        total_loss = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            batch_sender_emb = training_sender_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_data_rcv_emb = training_rcv_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_target = training_prediction[batch_idx*batch_size: (batch_idx+1)*batch_size].to(device)

            # move the sender and rcvr emb to the GPU




            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)

            loss = loss_function(batch_prediction.flatten(), batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            

        avg_loss = total_loss/total_batch
        if epoch_indx%epoch_interval == 0:
            print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))            
            # run validation
            # CHECK: if you use dropout layer, you might need to set some flag during inference step 
            validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(validation_set)
            # .to(device) to transfer to GPU
            batch_sender_emb = validation_sender_emb.to(device)
            batch_data_rcv_emb = validation_rcv_emb.to(device)
            batch_target = validation_prediction.to(device)
            
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            validation_loss = loss_function(batch_prediction.flatten(), batch_target)
            #print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))
            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_fusionMLP, args.model_name) #"model/my_model_fusionMLP.pickle")
                # model = torch.load("my_model.pickle")
                #torch.save(model_fusionMLP.state_dict(), "model/my_model_fusionMLP_state_dict.pickle")
                # model = nn.Sequential(...)
                # model.load_state_dict(torch.load("my_model.pickle"))
                print('*** min loss found! %g***'%validation_loss)


            #########
            batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
    
            for i in range(0, len(batch_prediction)):
                if batch_prediction[i]>= threshold_score:
                    batch_prediction[i] = 1
                else:
                    batch_prediction[i] = 0


            pred_class = batch_prediction

            TP = TN = FN = FP = 0
            P = N = 0
            for i in range (0, len(val_class)):
                if val_class[i] == 1 and pred_class[i] == 1:
                    TP = TP + 1
                    P = P + 1
                elif val_class[i] == 1 and pred_class[i] == 0:
                    FN = FN + 1
                    P = P + 1
                elif val_class[i] == 0 and pred_class[i] == 1:
                    FP = FP + 1
                    N = N + 1
                elif val_class[i] == 0 and pred_class[i] == 0:
                    TN = TN + 1
                    N = N + 1

            #print("P %d"%P)
            print('TP/P = %g, TN/N=%g '%(TP/P, TN/N))


            ########

            loss_curve[loss_curve_counter][0] = avg_loss
            loss_curve[loss_curve_counter][1] = validation_loss

            loss_curve_counter = loss_curve_counter + 1

            logfile=open(args.model_name+'_loss_curve.csv', 'wb')
            np.savetxt(logfile,loss_curve, delimiter=',')
            logfile.close()

def val_fusionMLP(val_set, model_name, threshold_score = 0.7):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    """
    model_fusionMLP = fusionMLP(
                 input_size = 512 + 264 + 1024, 
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
    
    validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(val_set)
    batch_sender_emb = validation_sender_emb.to(device)
    batch_data_rcv_emb = validation_rcv_emb.to(device)
    batch_target = validation_prediction.to(device)  
    batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
    batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
    prediction_class = []
    for i in range(0, len(batch_prediction)):
        if batch_prediction[i]>= threshold_score:
            prediction_class.append(1)
        else:
            prediction_class.append(0)

    return prediction_class, batch_prediction


def val_fusionMLP_multiBatch(dataset, model_name, threshold_score = 0.7, total_batch = 1):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    """
    model_fusionMLP = fusionMLP(
                 input_size = 512 + 264 + 1024, 
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
    batch_size = val_set.shape[0]//total_batch
    
    batch_prediction_combined = []
    for batch_idx in range(0, total_batch):
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

    if (batch_idx+1)*batch_size < val_set.shape[0]-1:
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