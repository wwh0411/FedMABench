#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/output
model=qwen2-vl-7b-instruct
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
round=149
val_dataset=/ailab/user/wangwenhao/ms-swift-main/output/gt_val_200_v1.json
peft_list=(
  v7-20241219-094924
  v8-20241219-094924
  v9-20241219-094924
)

for i in ${peft_list[@]};
do
    echo $i
#    if [ ! -d "$base_path/$model/$i/global_lora_$round-merged" ]; then
#
#        echo "*** merge ***"
#        CUDA_VISIBLE_DEVICES=$1 swift merge_lora --ckpt_dir "$base_path/$model/$i/global_lora_$round" --merge_lora true --model_type $model
#
#    fi

    # inference
    # check already exsist
    jsonl_files=$(find "$base_path/$model/$i/global_lora_$round/infer_result" -type f -name "*.jsonl")

    if [ -z "$jsonl_files" ]; then
    MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=$1 swift infer --ckpt_dir "$base_path/$model/$i/global_lora_$round" \
      --val_dataset $val_dataset --model_type $model --model_id_or_path $model_id_or_path --sft_type lora
    fi

    # calculate acc
    jsonl_files=$(find "$base_path/$model/$i/global_lora_$round/infer_result" -type f -name "*.jsonl")
    for jsonl_file in $jsonl_files; do
    # Process each jsonl file here
    echo $jsonl_file
#    cd evaluation
    python eval_acc.py --data_path "$jsonl_file"
    done


done
