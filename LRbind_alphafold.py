# Written By 
# Fatema Tuz Zohora


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
    parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()

    df = pd.read_csv(args.file_name, sep="\t")

    lrp_list = [[CCL21, CXCR4], [HLA-C, CXCR4], [HLA-A, CXCR4], [HLA-E, CXCR4], [HLA-B, CXCR4], [APOE, CXCR4], [HSP90B1, CXCR4], [HLA-F, CXCR4], [CCL19, CXCR4], [HLA-DRA, CD74]]
