import numpy as np
import h5py
print('package loading')
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import gzip
import argparse



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================ Specify data type firstly ===============
    parser.add_argument( '--file_name', type=str, default='uniprotkb_reviewed_true_AND_proteome_up_2025_02_27.tsv', help='The name of DB')
    #parser.add_argument( '--negatome_database_path', type=str, default='database/combined_stringent.txt', help='The name of DB')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv', help='The name of DB')
    #parser.add_argument( '--database_path_omnipath', type=str, default='database/omnipath_lr_db.csv', help='The name of DB')
    #parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()

    ### build a dictionary with key = gene, value = uniprotID
    df = pd.read_csv(args.file_name, sep="\t")
    dict_genes_uniprotID = defaultdict(list)
    for i in range (0, df['Sequence'].shape[0]):
        uniprotID = df['Entry'][i]
        if isinstance(df['Gene Names'][i], str):          
            gene_list = (df['Gene Names'][i]).split(' ')
            for gene in gene_list:
                dict_genes_uniprotID[gene].append(uniprotID)
            

    gene_remove = [] # remove genes having multiple uniprot id
    for gene in dict_genes_uniprotID:
        if len(dict_genes_uniprotID[gene]) > 1:
            print('multiple uniprot id found for gene '+ gene)
            #gene_remove.append(gene)

    
    df = pd.read_csv(args.database_path, sep=",")
    #lr_unique = defaultdict(dict)
    ligand_list = []
    receptor_list = []
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        #lr_unique[ligand][receptor] = 1
        ligand_list.append(ligand)
        receptor_list.append(receptor)



    with gzip.open('database/negatome_ligand_receptor_set', 'rb') as fp:  
    	negatome_ligand_list, negatome_receptor_list, lr_unique = pickle.load(fp)


    for ligand in negatome_ligand_list:
        ligand_list.append(ligand)

    for receptor in negatome_receptor_list:
        receptor_list.append(receptor)

    
    ligand_list = np.unique(ligand_list)
    receptor_list = np.unique(receptor_list)


    uniprot_vs_lr = defaultdict(list)
    for gene in ligand_list:
        #if gene in gene_remove:
        #    print('remove ligand ' + gene)
        #    continue
        if gene in dict_genes_uniprotID:    
            uniprot_vs_lr[dict_genes_uniprotID[gene][0]].append(gene)

    for gene in receptor_list:
        #if gene in gene_remove:
        #    print('remove rec ' + gene)
        #    continue
        if gene in dict_genes_uniprotID:
            uniprot_vs_lr[dict_genes_uniprotID[gene][0]].append(gene)


    for uniprot_id in uniprot_vs_lr:
        uniprot_vs_lr[uniprot_id] = list(np.unique(uniprot_vs_lr[uniprot_id]))
        if len(uniprot_vs_lr[uniprot_id]) > 1:
            
            print(uniprot_id + 'has multiple genes: ')
            print(uniprot_vs_lr[uniprot_id])
    
    gene_vs_embedding = dict()
    with h5py.File("per-protein.h5", "r") as file:
        print(f"number of entries: {len(file.items())}")
        for sequence_id, embedding in file.items():
            if sequence_id in uniprot_vs_lr: # and len(uniprot_vs_lr[sequence_id])==1:
                gene_vs_embedding[uniprot_vs_lr[sequence_id][0]] = np.array(embedding)
                print(
                    f"  id: {sequence_id}, "
                    f"  embeddings shape: {embedding.shape}, "
                    f"  embeddings mean: {np.array(embedding).mean()}"
                )

    print('len is %d'%len(gene_vs_embedding.keys()))
    with gzip.open('database/ligand_receptor_protein_embedding.pkl', 'wb') as fp:  
    	pickle.dump(gene_vs_embedding, fp)


  
