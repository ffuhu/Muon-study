#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=350m_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-10:00:00
#SBATCH --output=./logs/1b_test.out

# module purge
# module load 2023
# Your job starts in the directory where you call sbatch
# cd ../
# Activate your environment
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"

# Check whether the GPU is available
srun python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"
#for rank in 128 #256 512
#do

pairs=(
  "0.005 0.1"
  "0.005 0.25"
  "0.001 0.5"
  "0.001 0.75"
  "0.001 0.9"
  "0.001 1.0"
)

save_dir_base=/scratch-shared/HTJ2/checkpoints/new0
# Create a unique save directory by appending the SLURM job ID
save_dir="${save_dir_base}_${SLURM_JOB_ID}"
# Loop through each pair
#for pair in "${pairs[@]}"; do
#    read -r lr rank <<< "$pair"
for prj in 150
do
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr 2e-3 \
    --density 1.0 \
    --update_proj_gap 500 \
    --batch_size 64  \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --threshold 5000 \
    --save_dir $save_dir \
    --optimizer SPAM \
    --warmup_epoch $prj \
done

# done
# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"

