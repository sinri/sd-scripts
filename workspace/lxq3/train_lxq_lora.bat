echo NOT set TRAIN_MODEL_NAME=stable-diffusion-v1-5-LeXiaoQi-lora-6
echo NOT set PRETRAINED_MODE=E:\sinri\stable-diffusion-webui\models\Stable-diffusion\stable-diffusion-v1-5

set TRAIN_MODEL_NAME=anything-v5-LeXiaoQi-lora-1
set PRETRAINED_MODE=E:\OneDrive\Leqee\ai\AnythingV5

set TRAIN_MODEL_FORMAT=safetensors
accelerate launch --num_cpu_threads_per_process 1 ^
    E:\sinri\sd-scripts\train_network.py ^
    --pretrained_model_name_or_path=%PRETRAINED_MODE% ^
    --dataset_config=E:\sinri\sd-scripts\workspace\lxq3\lxq.toml ^
    --output_dir=E:\sinri\sd-scripts\workspace\lxq3\trained_model ^
    --output_name=%TRAIN_MODEL_NAME% ^
    --save_model_as=%TRAIN_MODEL_FORMAT% ^
    --prior_loss_weight=1.0 ^
    --max_train_steps=1000 ^
    --learning_rate=1e-4 ^
    --optimizer_type="AdamW8bit" ^
    --xformers ^
    --mixed_precision="fp16" ^
    --cache_latents ^
    --gradient_checkpointing ^
    --save_every_n_epochs=1  ^
    --network_module=networks.lora ^
