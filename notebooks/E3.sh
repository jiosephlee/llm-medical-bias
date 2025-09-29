python ESI_Finetuning.py \
    --dataset handbook \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

python ESI_Finetuning.py \
    --dataset ktas \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

# should take 2 hours