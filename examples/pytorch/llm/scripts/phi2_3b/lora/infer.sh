# Experimental environment: A10
# 8GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/phi2-3b/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --merge_lora false \
