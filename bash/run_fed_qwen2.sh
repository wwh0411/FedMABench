CUDA_VISIBLE_DEVICES=$2 MAX_PIXELS=602112 \
  swift sft \
  --round 30 \
  --fed_alg fedavg\
  --client_num 30 \
  --client_sample 5 \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path /ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct \
  --check_model_is_latest False \
  --lazy_tokenize True \
  --preprocess_num_proc 4 \
  --dataset $1 \
  --sft_type lora \
  --tuner_backend peft \
  --dtype AUTO \
  --output_dir output \
  --train_dataset_sample -1 \
  --dataset_test_ratio 0 \
  --max_steps -1 \
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


#  --output_dir $3 \
#  --add_output_dir_suffix False \
#  --custom_train_dataset_path /GPFS/data/wenhaowang-1/ms-swift/androidcontrol_1108/unpack-1109-test-message-vlm-train.jsonl \
