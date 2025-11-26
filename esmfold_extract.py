import sys

# Clear sys.argv to prevent argparse from parsing Colab-specific arguments
sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument( '--model_location', type=str,default='esm2_t33_650M_UR50D') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
parser.add_argument( '--fasta_file', type=str, default='APOE.fasta' , help='') #,required=True)
parser.add_argument( '--output_dir', type=pathlib.Path, default='APOE_emb_esm2', help='Name of the dataset')
parser.add_argument( '--nogpu', default=False, help='Name of the dataset')
parser.add_argument( '--toks_per_batch', type=int, default=4096, help='Name of the dataset')
parser.add_argument( '--truncation_seq_length', type=int, default=1022, help='Name of the dataset')
parser.add_argument( '--repr_layers', type=int, default=[-1], help='Name of the dataset')
parser.add_argument( '--include', type=str, default=['mean'], help='Name of the dataset')
parser
args = parser.parse_args()

run(args)


import torch
import os

# Assuming the output directory is 'APOE_emb_esm2' and the fasta file had one sequence
# named 'APOE', the file should be located at APOE_emb_esm2/APOE.pt

output_file_path = 'APOE_emb_esm2/APOE.pt'

if os.path.exists(output_file_path):
    loaded_result = torch.load(output_file_path)
    print("Contents of the .pt file:")
    print(loaded_result)
    print(loaded_result['mean_representations'][33])
    print(loaded_result['mean_representations'][33].shape)

    # Convert the tensor to a NumPy array
    numpy_array = loaded_result['mean_representations'][33].numpy()
    print("\nConverted to NumPy array:")
    print(numpy_array)
    print(numpy_array.shape)

else:
    print(f"File not found: {output_file_path}")