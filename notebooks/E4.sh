python ESI_Finetuning.py \
    --dataset handbook \
    --model_name unsloth/Qwen2.5-7B \
    --device_batch_size 4

python ESI_Finetuning.py \
    --dataset ktas \
    --model_name unsloth/Qwen2.5-7B \
    --device_batch_size 4

python ESI_Finetuning.py \
    --dataset mimic \
    --model_name unsloth/Qwen2.5-7B \
    --device_batch_size 4

# Should take 3 hours
