import numpy as np
import h5py
print('package loading')
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import gzip
import argparse
import torch
from pathlib import Path

def list_files_pathlib(directory_path):
    """Lists all files in a given directory using pathlib."""
    path_obj = Path(directory_path)
    files = [entry.name for entry in path_obj.iterdir() if entry.is_file()]
    return files



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================ Specify data type firstly ===============
    parser.add_argument( '--dir_name', type=str, default='database/ESMFold/esmFold_feature_vector_lastLayer/', help='path to the dir having ESMFold structure embedding')
    parser.add_argument( '--emb_dimension', type=int, default=1280, help='Dimension of ESMFold structure embedding')
    args = parser.parse_args()


    file_list = list_files_pathlib(args.dir_name)
    gene_vs_embedding = dict()
    global_min = []
    
    for file_name in file_list:
        gene_name = file_name.split('.')[0]

        loaded_result = torch.load(args.dir_name + file_name)
        #print("Contents of the .pt file:")
        #print(loaded_result)
        #print(loaded_result['mean_representations'][33])
        #print(loaded_result['mean_representations'][33].shape)

        # Convert the tensor to a NumPy array
        numpy_array = loaded_result['mean_representations'][33].numpy()
        #print("\nConverted to NumPy array:")
        #print(numpy_array)
        #print(numpy_array.shape)
        min_v = np.min(numpy_array)
        max_v = np.max(numpy_array)
        minmax_v_i = (numpy_array - min_v) / (max_v - min_v)

        
        print('min %g, max %g --> min %g, max %g'%(np.min(numpy_array), np.max(numpy_array), np.min(minmax_v_i), np.max(minmax_v_i)))
        gene_vs_embedding[gene_name] = minmax_v_i



    print('len is %d'%len(gene_vs_embedding.keys()))
    with gzip.open('database/esmFold_protein_structure_embedding.pkl', 'wb') as fp:  
    	pickle.dump(gene_vs_embedding, fp)

    print('all done')
    

  
