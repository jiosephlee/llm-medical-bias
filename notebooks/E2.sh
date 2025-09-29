python ESI_Finetuning.py \
    --dataset handbook \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

python ESI_Finetuning.py \
    --dataset handbook \
    --cpt \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

python ESI_Finetuning.py \
    --dataset handbook \
    --cpt \
    --para 5 \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

python ESI_Finetuning.py \
    --dataset handbook \
    --cpt \
    --para 10 \
    --model_name unsloth/Qwen2.5-14B \
    --device_batch_size 1

# This should take 20 minutes