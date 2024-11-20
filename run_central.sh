CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --round 200 \
  --fed_alg central \
  --model_type qwen-vl-chat \
  --dataset /GPFS/data/wenhaowang-1/ms-swift/androidcontrol_1108/unpack-1109-message-vlm-train-6000.jsonl \
  --sft_type lora \
  --tuner_backend peft \
  --dtype AUTO \
  --output_dir output \
  --train_dataset_sample 6000 \
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
