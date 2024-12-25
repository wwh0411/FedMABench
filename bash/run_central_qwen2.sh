CUDA_VISIBLE_DEVICES=2,3 MAX_PIXELS=602112 \
  swift sft \
  --round 200 \
  --fed_alg central \
  --client_num 1 \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path /ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct \
  --check_model_is_latest False \
  --lazy_tokenize True \
  --preprocess_num_proc 4 \
  --dataset /ailab/user/wangwenhao/ms-swift-main/output/high/alg5_train_5000_v1.json \
  --sft_type lora \
  --tuner_backend peft \
  --dtype AUTO \
  --output_dir output \
  --train_dataset_sample 60000 \
  --dataset_test_ratio 0 \
  --max_steps 30 \
  --max_length 4096 \
  --check_dataset_strategy warning \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing true \
  --batch_size 1 \
  --weight_decay 0.1 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 4 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.03 \
  --eval_strategy no \
  --save_strategy no \
  --logging_steps 100

#  --custom_train_dataset_path /GPFS/data/wenhaowang-1/ms-swift/androidcontrol_1108/unpack-1109-test-message-vlm-train.jsonl \
