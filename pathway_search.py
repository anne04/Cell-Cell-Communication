import numpy as np
import csv
import pickle
from scipy import sparse
import pandas as pd
from collections import defaultdict
species = 'Human'
receptor = ''
max_hop = 3

def get_KG(receptor_name, pathways_dict, max_hop){


}




pathways = pd.read_csv("pathways.csv")
pathways = pathways.drop(columns=[pathways.columns[0], pathways.columns[3],pathways.columns[4],pathways.columns[5]])
# keep only target species
for i in range (0, len(pathways)):
    if (pathways['species'][i]==species):
        pathways.drop([i])

pathways_dict = defaultdict(list)
for i in range (0, len(pathways)):
    if (pathways['species'][i]==species):
        pathways_dict[pathways['src'][i]].append([pathways['dest'][i], pathways['src_tf'][i], pathways['dest_tf'][i]])
