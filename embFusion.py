import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from embFusion_train_util import split_branch, shuffle_data#, data_to_tensor

#cellNEST_dimension = 512
geneEmb_dimension = 256 
proteinEmb_dimension = 1024

    
class fusionMLP(torch.nn.Module):
    def __init__(self, 
                 input_size:int = geneEmb_dimension + proteinEmb_dimension,  
                 hidden_size_fusion:int = 512, #512, 
                 output_size_fusion:int = 256, #256, 
                 hidden_size_predictor_layer1:int = 256*2, #256*2, 
                 hidden_size_predictor_layer2:int = 128 
                ):
        """
        This will initialize the model and return it.
        Args:
        input_size: concat size of gene embedding & protein embedding for a lig/rec gene
        hidden_size_fusion and output_size_fusion: hidden/output layer dimensions for emb fusion
        hidden_size_predictor_layer1 & hidden_size_predictor_layer2: hidden layers for ppi pred
        """
        super().__init__() # error happens without this 
        # Branch: sender
        self.sender_fusion_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size_fusion),
          nn.BatchNorm1d(hidden_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(hidden_size_fusion, output_size_fusion),
          nn.BatchNorm1d(output_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5)
        )

        # Branch: rcvr
        self.rcvr_fusion_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size_fusion),
          nn.BatchNorm1d(hidden_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(hidden_size_fusion, output_size_fusion),
          nn.BatchNorm1d(output_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5)
        )

        # concat both and pass through another MLP
        self.ppi_predict_layer = nn.Sequential(
            nn.Linear(output_size_fusion*2, hidden_size_predictor_layer1),
            nn.BatchNorm1d(hidden_size_predictor_layer1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_predictor_layer1, hidden_size_predictor_layer2),
            nn.BatchNorm1d(hidden_size_predictor_layer2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_predictor_layer2, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                sender_emb: torch.Tensor, 
                receiver_emb: torch.Tensor)-> torch.Tensor:
        
        fused_emb_sender = self.sender_fusion_layer(sender_emb)
        fused_emb_rcvr = self.rcvr_fusion_layer(receiver_emb)

        fused_emb_sender = F.normalize(fused_emb_sender, p=2) 
        fused_emb_rcvr = F.normalize(fused_emb_rcvr, p=2) 

        concat_fused_emb = torch.cat((fused_emb_sender, fused_emb_rcvr), dim=1)
        ppi_prediction = self.ppi_predict_layer(concat_fused_emb)
        return ppi_prediction




def train_fusionMLP(
    args,
    training_set: torch.Tensor,
    validation_set: torch.Tensor, 
    epoch = 2000,
    batch_size = 32,
    learning_rate =  1e-4,
    val_class = None,
    threshold_score = 0.7
    ):
    """
    args:
    training_set: torch.Tensor of training samples (80%)
    validation_set: torch.Tensor of validation samples (20%)
    val_class: list() of validation samples but with binary label (0/1)
    threshold_score: some cutoff to set binary labels
    """
    
    # CHECK = set a manual seed?
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    model_fusionMLP = fusionMLP().to(device) # CHECK = set the dimensions based on args.
    #model_fusionMLP = torch.load(args.model_name).to(device)

    
    print(model_fusionMLP)
    # set the loss function
    loss_function = nn.CrossEntropyLoss() #nn.MSELoss()

    # set optimizer
    optimizer = torch.optim.Adam(model_fusionMLP.parameters(), lr=learning_rate)
    epoch_interval = 20 # CHECK
    #### for plotting loss curve ########
    loss_curve = np.zeros((epoch//epoch_interval+1, 4))
    loss_curve_counter = 0
    ######################################
    total_training_samples = training_set.shape[0]
    total_batch = total_training_samples//batch_size
    min_loss = 10000 # just a big number to initialize
    for epoch_indx in range (0, epoch):
        # shuffle the training set
        training_sender_emb, training_rcv_emb, training_prediction = shuffle_data(training_set)        
        model_fusionMLP.train() # training mode
        total_loss = 0
        TP = TN = FN = FP = 0
        P = N = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            # get the batch of the sender and rcvr emb and move to GPU
            batch_sender_emb = training_sender_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_data_rcv_emb = training_rcv_emb[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            batch_target = training_prediction[batch_idx*batch_size: (batch_idx+1)*batch_size].to(device)
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)

            if batch_idx == 0 and epoch_indx%epoch_interval == 0:
                print('training:')
                print(batch_target[0:10])
                print(list(batch_prediction.flatten().cpu().detach().numpy())[0:10])

            loss = loss_function(batch_prediction.flatten(), batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # TP and TN rate
            batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
            val_class = 
            for i in range(0, len(batch_prediction)):
                if batch_prediction[i]>= threshold_score:
                    batch_prediction[i] = 1
                else:
                    batch_prediction[i] = 0

            
            pred_class = batch_prediction

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






             
        avg_loss = total_loss/total_batch
        if epoch_indx%epoch_interval == 0:
            #print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))            
            # run validation
            # CHECK: if you use dropout layer, you might need to set some flag during inference step 
            validation_sender_emb, validation_rcv_emb, validation_prediction = split_branch(validation_set)
            # .to(device) to transfer to GPU
            batch_sender_emb = validation_sender_emb.to(device)
            batch_data_rcv_emb = validation_rcv_emb.to(device)
            batch_target = validation_prediction.to(device)
            model_fusionMLP.eval()
            batch_prediction = model_fusionMLP(batch_sender_emb, batch_data_rcv_emb)
            validation_loss = loss_function(batch_prediction.flatten(), batch_target)
            if epoch_indx==0:
                min_loss = validation_loss
            print('Epoch %d/%d, Training loss: %g, val loss: %g'%(epoch_indx, epoch, avg_loss, validation_loss))

            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_fusionMLP, args.model_name)  
                # model = torch.load("my_model.pickle")
                #
                # torch.save(model_fusionMLP.state_dict(), "model/my_model_fusionMLP_state_dict.pickle")
                # model = nn.Sequential(...)
                # model.load_state_dict(torch.load("my_model.pickle"))
                print('*** min loss found! %g***'%validation_loss)


            #########
            batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
            print(batch_prediction[0:10])
            print(batch_target[0:10])
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


            ######## update the loss curve #########
            loss_curve[loss_curve_counter][0] = avg_loss
            loss_curve[loss_curve_counter][1] = validation_loss
            loss_curve[loss_curve_counter][2] = TP/P
            loss_curve[loss_curve_counter][3] = TN/N

            loss_curve_counter = loss_curve_counter + 1
            logfile=open(args.model_name+'_loss_curve.csv', 'wb')
            np.savetxt(logfile,loss_curve, delimiter=',')
            logfile.close()
            ############################

# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/