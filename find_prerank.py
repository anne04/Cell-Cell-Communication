import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group
import qnorm
#from sklearn.preprocessing import quantile_transform
import pickle
from scipy import sparse
import pickle
import scipy.linalg
from sklearn.metrics.pairwise import euclidean_distances
import gseapy as gp
from gseapy import gseaplot
import csv
import stlearn as st
from collections import defaultdict
####################  get the whole training dataset


#rootPath = os.path.dirname(sys.path[0])
#os.chdir(rootPath+'/CCST')

print("hello world!")

#if __name__ == "__main__":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
args = parser.parse_args()

#    main(args)
def processFile(f):

    df = pd.read_csv(f)
    df = df.rename(columns={"symbol": "feature"})

    # Replace infinite updated data with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    df.dropna(subset=["qVal", "pVal", "log2FC"], inplace=True)

    # df = df[df["qVal"] < 0.05]
    # df = df[(df["log2FC"]).abs() > 1]
    df["sample"] = f

    # Get maximum fold change values, one per feature.
    resDf = df.sort_values("log2FC", ascending=False).drop_duplicates(["feature"])

    return resDf

def main(args):
    print("hello world! main") 
    node_list = [86, 59, 48, 10, 13]
    name_list = ['10_13_48_59_86', '10_13_48_86_59', '10_13_59_86_48', '13_48_59_86_10', '10_48_59_86_13']
    n_index = 2
    #for n_index in range (0, len(name_list)):
        #toomany_label_file = '/cluster/home/t116508uhn/64630/differential_TAGConv_test_r4_14_15_org_whitelist.csv'
        toomany_label_file = '/cluster/home/t116508uhn/64630/differential_TAGConv_test_r4_'+name_list[n_index]+'_prerank.csv'
        #print(node_list[n_index])
        gene_dict=defaultdict(list)
        
        
        #dfs = processFile(toomany_label_file )
        
        
        with open(toomany_label_file) as file:
            csv_file = csv.reader(file, delimiter=",")
            i = 0
            for line in csv_file:
                if len(line)<1 or i == 0 or np.float64(line[1]) == np.inf or np.float64(line[1]) == -np.inf:
                    i = i+1
                    continue
                gene_dict[line[0]].append(np.float64(line[1]))
                i = i+1

        for gene in gene_dict:
            gene_dict[gene]=np.mean(gene_dict[gene])

        #signature_file='/cluster/home/t116508uhn/64630/Geneset_22Sep21_Subtypesonly.csv' # 1406
        signature_file='/cluster/home/t116508uhn/64630/GeneList_KF_22Aug10.csv'
        signature_info=defaultdict(list)
        #barcode_info.append("")
        with open(signature_file) as file:
            tsv_file = csv.reader(file, delimiter=",")
            for line in tsv_file:
                if (line[0].find('Basal') > -1) or (line[0].find('Classical') > -1) :
                    signature_info[line[0]].append(line[1])

        signature_info=dict(signature_info)

        data_rnk=pd.DataFrame.from_dict(gene_dict, orient='index')

        pre_res = gp.prerank(rnk = data_rnk,
                             gene_sets = signature_info,
                             threads=4,
                             min_size=0,
                             max_size=5000,
                             permutation_num=1000, # reduce number to speed up testing
                             outdir=None, # don't write to disk
                             seed=6,
                             verbose=True, # see what's going on behind the scenes
                            )
        print(pre_res.res2d)
        #pre_res.res2d.to_csv('/cluster/home/t116508uhn/64630/'+'14_vs_15'+'_prerank_tagconv_test_r4.csv')
        pre_res.res2d.to_csv('/cluster/home/t116508uhn/64630/'+str(node_list[n_index])+'_prerank_tagconv_test_r4.csv')

        terms = pre_res.res2d.Term
        for i in range (0, pre_res.res2d.shape[0]):
            # save figure
            # gseaplot(rank_metric=pre_res.ranking, term=terms[i], ofname=save_path+name_str+'_'+str(i)+'_prerank_tagconv_test_r4.svg', **pre_res.results[terms[i]])
            gseaplot(rank_metric=pre_res.ranking, term=terms[i], ofname='/cluster/home/t116508uhn/64630/'+str(node_list[n_index])+'_'+str(i)+'_prerank_tagconv_test_r4.svg', **pre_res.results[terms[i]])
            #gseaplot(rank_metric=pre_res.ranking, term=terms[i], ofname='/cluster/home/t116508uhn/64630/'+"14_vs_15"+'_'+str(i)+'_prerank_tagconv_test_r4.svg', **pre_res.results[terms[i]])


      



        
    
