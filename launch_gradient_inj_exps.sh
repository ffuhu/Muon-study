export LOCAL_RANK=0;
export MASTER_ADDR="localhost";
export MASTER_PORT=12355;
export RANK=0;
export WORLD_SIZE=1


# llama 60
python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 0.0005 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step -1 \
  --grad_injection_factor -1 \
  --grad_injection_elements -1 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "none" \
  --grad_injection_duration -1 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 0.0005 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.01 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 0.0005 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.05 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns


# llama 130
python torchrun_main1_test.py --model_config configs/llama_130m.json \
  --lr 8e-4 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step -1 \
  --grad_injection_factor -1 \
  --grad_injection_elements -1 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "none" \
  --grad_injection_duration -1 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_130m.json \
  --lr 8e-4 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.01 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_130m.json \
  --lr 8e-4 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.05 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

# llama 1B
python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 2e-3 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step -1 \
  --grad_injection_factor -1 \
  --grad_injection_elements -1 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "none" \
  --grad_injection_duration -1 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 2e-3 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.01 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns

python torchrun_main1_test.py --model_config configs/llama_60m.json \
  --lr 2e-3 \
  --batch_size 64 \
  --total_batch_size 512 \
  --num_training_steps 10000 \
  --warmup_steps 150 \
  --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --optimizer muongradientinjection \
  --model_type llama \
  --data_type c4 \
  --single_gpu \
  --grad_injection_step 200 \
  --grad_injection_factor 0.01 \
  --grad_injection_elements 0.05 \
  --grad_injection_layer_number -1  \
  --grad_injection_fn "step" \
  --grad_injection_duration 5 \
  --save_gradients_every 50 \
  --finish_execution_at 1000 \
  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
  --grad_norm_scaling \
  --exp_name gns