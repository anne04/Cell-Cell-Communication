  GNU nano 4.6                                                                                                                                   gpu_threaded_job.sh                                                                                                                                   Modified  
#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=schwartzgroup_gpu
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH -c=12  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=30GB        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH --time=10:00:00
#SBATCH --job-name=fatema_test1
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
source /cluster/home/t116508uhn/env_CCC_cpu/bin/activate
cd /cluster/projects/schwartzgroup/fatema/CCST/

# load necessary modules
module load python3
module load pytorch_gpu

# run your python script with parameters
python run_CCC_gat.py --data_type nsc --data_name V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new --lambda_I 0.8 --DGI 1 --data_path /cluster/projects/schwartzgroup/fatema/find_ccc/ --model_path new_alignment/model_ccc_rgcn/ --embedding_data_path new_alignment/Embedding_data_ccc_rgcn/ --result_path new_alignment/result_ccc_rgcn/ --num_epoch 80000 --workflow_v=1 --hidden 512 --cluster 0 --retrain 0 --all_distance 0 --meu 0.2 --model_name 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_r2' --heads 1 --training_data 'adjacency_records_synthetic_data_ccc_roc_control_model_dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --num_cells 3000 --options 'dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --withFeature 'no' --datatype='synthetic' > output_synthetic_data_ccc_roc_control_model_4_path_threshold_distance_cellcount3000_e_gatconv_3dim_r2.log &
python run_CCC_gat.py --data_type nsc --data_name V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new --lambda_I 0.8 --DGI 1 --data_path /cluster/projects/schwartzgroup/fatema/find_ccc/ --model_path new_alignment/model_ccc_rgcn/ --embedding_data_path new_alignment/Embedding_data_ccc_rgcn/ --result_path new_alignment/result_ccc_rgcn/ --num_epoch 80000 --workflow_v=1 --hidden 512 --cluster 0 --retrain 0 --all_distance 0 --meu 0.2 --model_name 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_r3' --heads 1 --training_data 'adjacency_records_synthetic_data_ccc_roc_control_model_dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --num_cells 3000 --options 'dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --withFeature 'no' --datatype='synthetic' > output_synthetic_data_ccc_roc_control_model_4_path_threshold_distance_cellcount3000_e_gatconv_3dim_r3.log &
python run_CCC_gat.py --data_type nsc --data_name V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new --lambda_I 0.8 --DGI 1 --data_path /cluster/projects/schwartzgroup/fatema/find_ccc/ --model_path new_alignment/model_ccc_rgcn/ --embedding_data_path new_alignment/Embedding_data_ccc_rgcn/ --result_path new_alignment/result_ccc_rgcn/ --num_epoch 80000 --workflow_v=1 --hidden 512 --cluster 0 --retrain 0 --all_distance 0 --meu 0.2 --model_name 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_r4' --heads 1 --training_data 'adjacency_records_synthetic_data_ccc_roc_control_model_dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --num_cells 3000 --options 'dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --withFeature 'no' --datatype='synthetic' > output_synthetic_data_ccc_roc_control_model_4_path_threshold_distance_cellcount3000_e_gatconv_3dim_r4.log &
python run_CCC_gat.py --data_type nsc --data_name V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new --lambda_I 0.8 --DGI 1 --data_path /cluster/projects/schwartzgroup/fatema/find_ccc/ --model_path new_alignment/model_ccc_rgcn/ --embedding_data_path new_alignment/Embedding_data_ccc_rgcn/ --result_path new_alignment/result_ccc_rgcn/ --num_epoch 80000 --workflow_v=1 --hidden 512 --cluster 0 --retrain 0 --all_distance 0 --meu 0.2 --model_name 'synthetic_data_ccc_roc_control_model_4_path_threshold_distance_e_gatconv_3dim_r5' --heads 1 --training_data 'adjacency_records_synthetic_data_ccc_roc_control_model_dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --num_cells 3000 --options 'dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim' --withFeature 'no' --datatype='synthetic' > output_synthetic_data_ccc_roc_control_model_4_path_threshold_distance_cellcount3000_e_gatconv_3dim_r5.log &

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
