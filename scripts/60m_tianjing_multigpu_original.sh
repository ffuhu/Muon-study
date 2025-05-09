#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=v2_c4_llama_60m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --output=./logs/v2_c4_llama_60m.out
# module purge
# module load 2023
# Your job starts in the directory where you call sbatch
cd ../
# Activate your environment
source activate sds
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"
# Check whether the GPU is available
srun python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"
#for rank in 128 #256 512
#do
#bfloat16
pairs=(
  "0.0005 1.0"
)
# Loop through each pair
for pair in "${pairs[@]}"; do
    read -r lr rank <<< "$pair"
    torchrun --standalone --nproc_per_node 1 torchrun_main1_test.py \
        --model_config configs/llama_60m.json \
        --lr $lr \
        --galore_scale 0.25 \
        --rank $rank \
        --update_proj_gap 200000 \
        --batch_size 64  \
        --total_batch_size 512 \
        --num_training_steps 10000 \
        --warmup_steps 1000 \
        --weight_decay 0 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer adamw \
        --proj_type std \
        --updating_mask_method random \
        --model_type llama \
        --data_type c4 \
        --trick none
done
# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"