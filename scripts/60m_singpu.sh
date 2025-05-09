#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=130m_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=./logs/130m_test.out

# module purge
# module load 2023
#source activate spam
# Your job starts in the directory where you call sbatch
# cd ../
# Activate your environment
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"

# Check whether the GPU is available
python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"
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

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export OMP_NUM_THREADS=8
export MASTER_ADDR="localhost"
export MASTER_PORT=12355
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0


save_dir_base=./checkpoints/
# Create a unique save directory by appending the SLURM job ID
save_dir="${save_dir_base}/test"
# Loop through each pair
#for pair in "${pairs[@]}"; do
#    read -r lr rank <<< "$pair"
for prj in 150
do
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 1e-3 \
    --density 1.0 \
    --update_gap 500 \
    --batch_size 64  \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --threshold 5000 \
    --save_dir $save_dir \
    --optimizer AdamWGradients \
    --warmup_epoch $prj \
    --single_gpu
done

# for prj in 2000 
# do
# torchrun --standalone --nproc_per_node 1 torchrun_main_sampling1_fuse.py \
#     --model_config configs/llama_60m.json \
#     --lr 6e-3 \
#     --galore_scale 0.25 \
#     --rank 1.0 \
#     --update_proj_gap 2000 \
#     --batch_size 128  \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --threshold 0 \
#     --save_dir $save_dir \
#     --optimizer galore_adamw \
#     --proj_type std \
#     --updating_mask_method random \
#     --zero_state \
#     --zero_all_state \
#     --warmup_epoch 300 \
#     --init_mask random \
#     --zero_grad \
#     --grad_accu_steps 0 \
#     --grad_clipping 1.0
# done

# done
# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"

