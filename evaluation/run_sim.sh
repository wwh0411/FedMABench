#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/output
#model=qwen2-vl-7b-instruct
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
model=internvl2-8b
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-8B

val_dataset=/ailab/user/wangwenhao/ms-swift/output/gt_val_200_v1.json
peft_list=(
  a
)

for i in ${peft_list[@]};
do
    echo $i

    # calculate acc
    jsonl_files=$(find "/ailab/user/wangwenhao/ms-swift/output/high" -type f -name "*.json")
    for jsonl_file in $jsonl_files; do
    # Process each jsonl file here
    echo $jsonl_file
#    cd evaluation
    python eval_gpt.py --data_path "$jsonl_file"
    done


done
