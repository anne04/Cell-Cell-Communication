import numpy as np
import csv
import pickle
from scipy import sparse
import pandas as pd

species = 'Human'
receptor = ''

def get_KG(){


}




pathways = pd.read_csv("pathways.csv")
df.drop(columns=['B', 'C'])
pathways.drop(columns=[pathways.columns[0], pathways.columns[3],pathways.columns[4],pathways.columns[5]])
# keep only target species
for i in range (0, len(pathways)):
    if (pathways['species'][i]!=species):
        pathways.drop([i])

