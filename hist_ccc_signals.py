# written by GW                                                                                                                                                                     /mnt/data0/gw/research/notta_pancreatic_cancer_visium/plots/fatema_signaling/hist.py                                                                                                                                                                                         
import sys

import altair as alt
import pandas as pd
import scipy.stats
import altairThemes

#sys.path.append("/home/gw/code/utility/altairThemes/")
#if True:  # In order to bypass isort when saving
#    import altairThemes

def readCsv(x):
  """Parse file."""
  #colNames = ["method", "benchmark", "start", "end", "time", "memory"]
  df = pd.read_csv(x, sep=",")

  return df

def preprocessDf(df):
  """Transform ligand and receptor columns."""
  df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
  df["component"] = df["component"].astype(str).str.zfill(2)

  return df

def statOrNan(xs, ys):
  if len(xs) == 0 or len(ys) == 0:
    return None
  else:
    return scipy.stats.mannwhitneyu(xs, ys)

def summarizeStats(df, feature):
  meanRes = df.groupby(["benchmark", "method"])[feature].mean()
  statRes = df.groupby("benchmark").apply(lambda x: post.posthoc_ttest(x, val_col = feature, group_col = "method", p_adjust = "fdr_bh"))

  return (meanRes, statRes)

def writeStats(stats, feature, outStatsPath):
  stats[0].to_csv(outStatsPath + "_feature_" + feature + "_mean.csv")
  stats[1].to_csv(outStatsPath + "_feature_" + feature + "_test.csv")

  return

def plot(df):

  set1 = altairThemes.get_colour_scheme("Set1", len(df["component"].unique()))

  base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale = alt.Scale(range=set1)),
            order=alt.Order("component", sort="ascending"),
            tooltip=["component"]
        )
  p = base

  return p

def totalPlot(df, features, outPath):

  p = alt.hconcat(*map(lambda x: plot(df, x), features))

  outPath = outPath + "_boxplot.html"

  p.save(outPath)

  return

def main():
  pd.set_option('display.max_columns', 50)
  # register the custom theme under a chosen name
  alt.themes.register("publishTheme", altairThemes.publishTheme)
  # enable the newly registered theme
  alt.themes.enable("publishTheme")
  #data_options = 'Female_Virgin_ParentingExcitatory_0.21_20'
  #inFile = '/cluster/home/t116508uhn/64630/ccc_th98_records' + data_options + '.csv'
  #inFile = '/cluster/home/t116508uhn/64630/ccc_th95_omnipath_records_withFeature_woBlankEdges.csv' #sys.argv[1]
  #inFile = '/cluster/home/t116508uhn/64630/ccc_th97_records_woBlankEdges_bothAbove98th.csv' #sys.argv[1]
  inFile = '/cluster/home/t116508uhn/64630/ccc_th97_records_woBlankEdges_bothAbove98th_97th_intersection.csv' #sys.argv[1]
  df = readCsv(inFile)
  df = preprocessDf(df)
  outPathRoot = inFile.split('.')[0]

  p = plot(df)
  #outPath = '/cluster/home/t116508uhn/64630/ccc_th98_hist_'+data_options+'_woBlankEdges.html' #outPathRoot + "_histogram.html"
  #outPath = '/cluster/home/t116508uhn/64630/ccc_th98_hist_woBlankEdges.html' #outPathRoot + "_histogram.html"
  outPath = '/cluster/home/t116508uhn/64630/test_hist.html' #outPathRoot + "_histogram.html"
  p.save(outPath)

  # outStatsPath = outPath + "_stats"
  # list(map(lambda feature: writeStats(summarizeStats(df, feature), feature, outStatsPath), features))

  return

if __name__ == "__main__":
  status = main()
  sys.exit(status)

