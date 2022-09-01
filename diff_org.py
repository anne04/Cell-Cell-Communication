  GNU nano 6.3                                                                   /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/diff.org                                                                    Modified  
* Input
- Node 10 :: Classical B
- Node 59 :: Basal B
- Node 14 :: Bottom right Classical A
- Node 15 :: Top right Classical A
- Node 13 :: Bottom and Top right Classical A

#+name: in-files-sig
- ./differential_analysis/differential_TAGConv_test_r4_13_59_org_whitelist.csv
# - ./differential_analysis/differential_TAGConv_test_r4_10_59_org_whitelist.csv
# - ./differential_analysis/differential_TAGConv_test_r4_14_15_org_whitelist.csv

* Intersection
#+name: find-intersection
#+begin_src python :results output :var inFilesSig=in-files-sig
  import os
  import glob
  import pandas as pd
  import shutil
  import numpy as np
  import sys
  import scikit_posthocs as post

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

  def processSignatureFile(f):

    df = pd.read_csv(f)

    resDf = df.rename(columns={"Name": "feature", "List": "signature"})

    return resDf

  def intersect(df, sDf):
    resDf = df.merge(sDf, how="inner", on="feature")[["feature", "log2FC", "pVal", "qVal", "signature", "sample"]]
    resDf.drop_duplicates(inplace=True)

    return resDf

  def summarizeSig(df):
    meanRes = df.groupby("signature")["log2FC"].mean()
    statDf = df[df["signature"].str.contains("Classical|Basal")]
    statRes = post.posthoc_mannwhitney(statDf, val_col = "log2FC", group_col = "signature", p_adjust = "fdr_bh")

    return (meanRes, statRes)

  def main():
    pd.set_option('display.max_columns', 50)

    signatureFile = "./GeneList_KF_22Aug10.csv"

    dfs = map(processFile, [x for xs in inFilesSig for x in xs])
    sDf = processSignatureFile(signatureFile)

    df = pd.concat(map(lambda x: intersect(x, sDf), dfs))
    resDf = summarizeSig(df)
    # resDf = df.sort_values("log2FC", ascending=False)

    print(resDf[0])
    print(resDf[1])

    # outFile = os.path.basename(inFilesSig[0][0]).split(".")[0] + "_signals.csv.gz"

    # resDf.to_csv(outFile, index = False)

    # print(outFile)

    return

  if __name__ == "__main__":
      status = main()
      sys.exit(status)
