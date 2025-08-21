import torch
import numpy as np

cellNEST_dimension = 512
geneEmb_dimension = 256 
proteinEmb_dimension = 1024


def split_branch(data: torch.Tensor) -> torch.Tensor, torch.Tensor, torch.Tensor:
    prediction_column = data.size(1)-1
    rcvr_dimension_total = sender_dimension_total = cellNEST_dimension + geneEmb_dimension + proteinEmb_dimension
    sender_emb = data[:, cellNEST_dimension:sender_dimension_total]    
    rcv_emb = data[:, sender_dimension_total+cellNEST_dimension:sender_dimension_total+rcvr_dimension_total]
    prediction = data[:, prediction_column]
    return sender_emb, rcv_emb, prediction 

def shuffle_data(
    training_set: torch.Tensor 
    ) -> torch.Tensor, torch.Tensor, torch.Tensor:
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
    return training_sender_emb, training_rcv_emb, training_prediction 
    
def data_to_tensor(
    training_set, #: list()
    remove_set=None
    ) -> torch.Tensor, list(np.array):
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