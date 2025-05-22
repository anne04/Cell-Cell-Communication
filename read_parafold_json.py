import pandas as pd
import json 
import glob
import matplotlib.pyplot as plt
import seaborn as sns


score_list = []
file_path = 'ParallelFold-main/output/lrbind_SERPING1_NCL_score.json'
file_list = glob.glob("ParallelFold-main/output/manual_*json")
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
