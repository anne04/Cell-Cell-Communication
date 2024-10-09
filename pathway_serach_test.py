import numpy as np
import csv
import pickle
from scipy import sparse
import pandas as pd
from collections import defaultdict
from collections import deque


species = 'Human'
receptor = '' 
max_hop = 3
def pathway_expression(receptor, get_rows, gene_exist_list, TF_genes, only_TF):
    table_info = filter_pathway(get_rows, gene_exist_list)
    adjacency_list = get_adjacency_list(table_info)
    TF_scores = get_bfs(adjacency_list, receptor, TF_genes)
    # take the weighted average of the TF
    score = 0


    
    for gene in TF_scores:
        hop = TF_scores[gene] 
        if only_TF == 1:
            if gene in TF_genes:
                score = score + gene_exist_list[TF_gene] # scores multiplied if available
                
        else:
            score = score + gene_exist_list[TF_gene] # scores multiplied if available
            
    return score

def get_bfs(adjacency_list, receptor, TF_genes):
    TF_scores = defaultdict(int)
    q = deque()
    total_TF = len(adjacency_list.keys())-1

    hop_count = 1
    for i in range (0, len(adjacency_list[receptor])):
        dest = adjacency_list[receptor][i][0]
        q.append(dest)
        TF_scores[dest]=hop_count
        
    while(len(TF_scores.keys())!= total_TF):
        source_gene = q.popleft()
        for i in range (0, len(adjacency_list[source_gene])):
            dest = adjacency_list[source_gene][i][0]
            if dest in TF_scores: # path already visited so go back
                continue
                
            q.append(dest)
            TF_scores[dest] =  TF_scores[source_gene] + 1 


    TF_found = 0
    for gene in TF_scores:
        if gene in TF_genes:
            TF_found = 1
            break
            
    if TF_found == 1:
        return TF_scores
    else:
        return 0

# get_rows is a table, each row is info on source and target
# get_rows is updated in each call
def get_KG(receptor_name, pathways_dict, max_hop, get_rows, current_hop):
    if  get_rows[len(get_rows)-1][1][2]== 'YES' or current_hop == max_hop: # run as long as you don't reach a TF or max_hop is reached. 
        return 

    for i in range (0, len(pathways_dict[receptor_name])):
        get_rows.append([receptor_name, pathways_dict[receptor_name][i]])
        print(pathways_dict[receptor_name][i])
        get_KG(pathways_dict[receptor_name][i][0], pathways_dict, max_hop, get_rows, current_hop+1)
    
    return 

def filter_pathway(table_info, gene_exist_list): # gene_exist_list is a dictionary
    # update table_info based on gene_exist_list
    get_rows = []
    for i in range (0, len(table_info)):
        if table_info[i][0] in gene_exist_list and table_info[i][1] in gene_exist_list:
            get_rows.append([table_info[i][0], table_info[i][1]])
    return table_info

def get_adjacency_list(table_info):
    adjacency_list = defaultdict(list)
    unique_gene_set = dict()   
    for i in range (0, len(table_info)):
        source = table_info[i][0]
        dest = table_info[i][1][0]
        adjacency_list[source].append(dest)
        unique_gene_set[source]=''
        unique_gene_set[dest]=''
    
    for gene in unique_gene_set:
        if gene not in adjacency_list:
            adjacency_list[gene]=[]
            
    return adjacency_list

'''
pathways = pd.read_csv("pathways.csv")
pathways = pathways.drop(columns=[pathways.columns[0]])
pathways.to_csv("pathways_NEST.csv", index=False) 
'''
pathways = pd.read_csv("pathways_NEST.csv")        
pathways = pathways.drop(columns=[pathways.columns[2],pathways.columns[3],pathways.columns[4]])
pathways = pathways.drop_duplicates(ignore_index=True)
# keep only target species
pathways_dict = defaultdict(list)
for i in range (0, len(pathways)):
    if (pathways['species'][i]==species):
        pathways_dict[pathways['src'][i]].append([pathways['dest'][i], pathways['src_tf'][i], pathways['dest_tf'][i]])

# filter pathway based on common genes in data set
# ...
# then make a kg for each receptor and save it somewhere
get_rows = []
get_KG('ERF', pathways_dict, 2, get_rows, current_hop=0) # save it

# for each cell, for each receptor
gene_exist_list = dict()
gene_exist_list[]=
# assign genes to gene exist list
receptor_expression = pathway_expression(get_rows, gene_exist_list) 
