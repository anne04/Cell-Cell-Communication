import pandas as pd
import json 
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

score_list = []
file_list = glob.glob("ParallelFold-main/output/manual*json")
for file_path in file_list:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    score_list.append(data['iptm+ptm']['model_1_multimer_v3_pred_0'])


plt.clf()
sns.histplot(score_list, bins=30, kde=True, color='skyblue') #kde adds a kernel density estimate
plt.title('AlphaFold score distribution for manually curated LR DB')
plt.xlabel('AlphaFold Score')
plt.ylabel('Frequency')
plt.savefig('/cluster/home/t116508uhn/LRbind_output/AF_score_distribution_manualLRP_'+str(len(score_list))+'LRP.jpg')


##############
output_path = '/cluster/home/t116508uhn/LRbind_output/'
plot_title = 'AlphaFold score distribution for predicted LRP' #random  
file_name = 'AF_score_distribution_predictedLRP_' #'AF_score_distribution_randomLRP_' #'AF_score_distribution_selfbindLRP_' #'AF_score_distribution_manualLRP_' # 'AF_score_distribution_falseLRP_' #
lr_type =  'lrbind' #'random' #'false' # #manual
lrpair_score_dict = defaultdict(list)
score_list = []
file_list = glob.glob("ParallelFold-main/output/"+ lr_type +"*json")
for file_path in file_list:
    if '_old_' in file_path:
        continue
    with open(file_path, 'r') as file:
        data = json.load(file)

    AF_score = data['iptm+ptm']['model_1_multimer_v3_pred_0']
    score_list.append(AF_score)
    ligand = file_path.split('_')[1]
    receptor = file_path.split('_')[2]
    lrpair_score_dict['pair'].append(ligand + '_to_' + receptor)
    lrpair_score_dict['AF scores'].append(AF_score)

plt.clf()
sns.histplot(score_list, bins=30, kde=True, color='skyblue') #kde adds a kernel density estimate
plt.title(plot_title) #predicted #selfbind
plt.xlabel('AlphaFold Score')
plt.ylabel('Frequency')
plt.savefig(output_path + file_name +str(len(score_list))+'LRP.jpg')

data_list_pd = pd.DataFrame(lrpair_score_dict)
data_list_pd.to_csv(output_path + file_name +str(len(score_list))+'LRP.csv' , index=False)
###################################################################################################
top_N = 100
ligand_list = []
receptor_list = [] #
data_name = ['LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir', 'LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir',\
	'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir']
model_list = ['model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L', 'model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L',\
	      'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L']
j = 0 # \
for j in range (0, len(model_list)):
model_name = model_list[j]
df = pd.read_csv('/cluster/home/t116508uhn/LRbind_output/'+ data_name[j] + '/' +model_name+'_down_up_deg_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelsOutOfallLR.csv', sep=",")
    
for i in range (35, 100):
    ligand = df["Ligand-Receptor Pairs"][i].split('+')[0]
    receptor = df["Ligand-Receptor Pairs"][i].split('+')[1]       
    ligand_list.append(ligand)
    receptor_list.append(receptor)
j = j + 1




probable_pairs = []
for i in range(0, len(ligand_list)):
probable_pairs.append([ligand_list[i], receptor_list[i]])



# Print the data
print(data)

{
    "iptm+ptm": {
        "model_1_multimer_v3_pred_0": 0.2626855865187953
    },
    "order": [
	"model_1_multimer_v3_pred_0"
    ]
}
