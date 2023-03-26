  GNU nano 4.6                                                                                                                                   gpu_threaded_job.sh                                                                                                                                   Modified  
#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-gregorys
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=63500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH --time=06:00:00
#SBATCH --job-name=fatema_test1
#SBATCH --output=some_name-%j.out
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# activate your virtual environment
source /home/fatema/ENV/bin/activate
# load necessary modules
module load python/3.10
# run your python script with parameters
python /project/def-gregorys/fatema/GCN_clustering/run_CCST_edited.py --data_name=exp2_V10M25_61_D1_64630_Spatial10X   --num_epoch=15000 --hidden=256 --model_name=exp2_V10M25_61_D1_64630_Spatial10X_test2 --GNN_type='TAGConv'

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
