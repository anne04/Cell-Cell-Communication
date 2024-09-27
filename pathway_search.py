import numpy as np
import csv
import pickle
from scipy import sparse
import pandas as pd
from collections import defaultdict
species = 'Human'
receptor = ''
max_hop = 3

def get_bfs(adjacency_list):
    TF_scores = dict()
    



# get_rows is a table, each row is info on source and target
# get_rows is updated in each call
def get_KG(receptor_name, pathways_dict, max_hop, get_rows, current_hop):
    if current_hop == max_hop:
        return 

    for i in range (0, len(pathways_dict[receptor_name])):
        get_rows.append([receptor_name, pathways_dict[receptor_name][i]])
        print(pathways_dict[receptor_name][i])
        get_KG(pathways_dict[receptor_name][i][0], pathways_dict, max_hop, get_rows, current_hop+1)
    
    return 

def get_adjacency_list(table_info):
    adjacency_list = defaultdict(list)
       
    for i in range (0 , len(table_info)):
        source = table_info[i][0]
        dest = table_info[i][1][0]
        adjacency_list[source].append(dest)

    return adjacency_list

pathways = pd.read_csv("pathways.csv")
pathways = pathways.drop(columns=[pathways.columns[0], pathways.columns[3],pathways.columns[4],pathways.columns[5]])
# keep only target species
pathways_dict = defaultdict(list)
for i in range (0, len(pathways)):
    if (pathways['species'][i]==species) and (pathways['src_tf'][i]=='YES' or pathways['dest_tf'][i]=='YES'):
        pathways_dict[pathways['src'][i]].append([pathways['dest'][i], pathways['src_tf'][i], pathways['dest_tf'][i]])

get_rows = []
get_KG('ERF', pathways_dict, 2, get_rows, current_hop=0)
