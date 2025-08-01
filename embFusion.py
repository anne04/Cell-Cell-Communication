import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class fusionMLP(torch.nn.Module):
    def __init__(self, 
                 input_size: int = 512 + 264 + 1023, 
                 hidden_size_fusion: int = 1024, 
                 output_size_fusion: int = 256,
                 hidden_size_predictor_layer1: int = 256*2,
                 hidden_size_predictor_layer2: int = 256
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
    dataset: list(),
    epoch: int = 1000,
    batch_size: int = 32
    ):
    
        
        

                   

      




# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
