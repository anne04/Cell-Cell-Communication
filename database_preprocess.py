import numpy as np
from collections import defaultdict
import pandas as pd




################# for running Niches ###############################
ligand_dict_dataset = defaultdict(list)
cell_cell_contact = dict() 
ligand_list = []
receptor_list = []
annotation_list = []   

cell_chat_file = '/cluster/home/t116508uhn/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]       
    if df["annotation"][i] == 'ECM-Receptor':   # since we are considering cell to cell communication 
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        ligand_dict_dataset[ligand].append(receptor)
        if df["annotation"][i] == 'Cell-Cell Contact':
            cell_cell_contact[receptor] = '' 
            annotation_list.append('Cell-Cell Contact')
        else:
            annotation_list.append('')
            
        ligand_list.append(ligand)
        receptor_list.append(receptor)
        
            
print(len(ligand_dict_dataset.keys()))

nichetalk_file = '/cluster/home/t116508uhn/NicheNet-LR-pairs.csv'   
df = pd.read_csv(nichetalk_file)
for i in range (0, df["from"].shape[0]):
    ligand = df["from"][i]
    receptor = df["to"][i]
    ligand_list.append(ligand)
    receptor_list.append(receptor)
    annotation_list.append('')

# make a csv file with three columns: Ligand, Receptor, Annotation
csv_record = []
csv_record.append(['Ligand', 'Receptor', 'Annotation'])
for i in range (0, len(ligand_list)):
    csv_record.append([ligand_list[i], receptor_list[i], annotation_list[i]])

df = pd.DataFrame(csv_record) # output 4
df.to_csv('/cluster/home/t116508uhn/64630/NEST_database.csv', index=False, header=False)


    