import numpy as np
import csv
import pickle
from scipy import sparse
import pandas as pd
from collections import defaultdict
from collections import deque


def pathway_expression(receptor, get_rows, gene_exist_list, TF_genes):
    table_info = filter_pathway(get_rows, gene_exist_list)
    adjacency_list = get_adjacency_list(table_info)
    intra_scores = get_bfs(adjacency_list, receptor, TF_genes)
    # take the weighted average of the TF
    score = 0
    weight = 0
    for gene in intra_scores:
        if gene in TF_genes:
            hop = intra_scores[gene] 
            score = score + gene_exist_list[gene]*(1/hop)
            weight = weight + (1/hop)
        
    return score/weight


def get_bfs(adjacency_list, receptor, TF_genes):
    TF_scores = defaultdict(int)
    q = deque()
    total_TF = len(adjacency_list.keys())-1

    hop_count = 1
    for i in range (0, len(adjacency_list[receptor])):
        dest = adjacency_list[receptor][i][0]
        q.append(dest)
        TF_scores[dest]=hop_count # [hop_count, score] -> use score if you have one

    # Also print BFS paths to see which paths contain TF. 
    # Keep nonTF genes of those paths only and ignore other nonTF genes
    while(len(TF_scores.keys())!= total_TF):
        source_gene = q.popleft()
        for i in range (0, len(adjacency_list[source_gene])):
            dest = adjacency_list[source_gene][i][0]
            if dest in TF_scores:
                continue
                
            q.append(dest)
            hop_count = TF_scores[source_gene] + 1
            TF_scores[dest] =  hop_count # [hop_count, score] -> use score if you have one
            
    TF_found = 0
    for gene in TF_scores:
        if gene in TF_genes:
            TF_found = 1
            break
            
    if TF_found == 1:
        return TF_scores
    else:
        return []    


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

