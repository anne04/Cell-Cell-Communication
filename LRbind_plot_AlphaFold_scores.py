import pandas as pd
import json 
import glob
import matplotlib.pyplot as plt
import seaborn as sns


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
file_name = 'AF_score_distribution_predictedLRP_' #'AF_score_distribution_manualLRP_' #
lr_type =  'false' #'lrbind' #'false' #'lrbind' #manual
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
plt.title('AlphaFold score distribution for predicted LRP')
plt.xlabel('AlphaFold Score')
plt.ylabel('Frequency')
plt.savefig(output_path + file_name +str(len(score_list))+'LRP.jpg')

data_list_pd = pd.DataFrame(lrpair_score_dict)
data_list_pd.to_csv(output_path + file_name +str(len(score_list))+'LRP.csv' , index=False)

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
