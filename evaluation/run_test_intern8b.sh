#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/bash/output
#model=qwen2-vl-7b-instruct
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
model=internvl2-8b
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-8B
round=20
val_dataset=/ailab/user/wangwenhao/ms-swift/output/gt_val_200_v1_low.json
peft_list=(
# high train
#v10-20241226-205433
#v11-20241226-205441
#v12-20250106-234557
#v13-20250106-234622
#v14-20250106-235056
#v21-20250107-171230
# low train
v15-20250106-235705
v16-20250106-235742
v17-20250106-235914
v18-20250107-000105
v19-20250107-000517
#v20-20250107-000526

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
    python test_swift.py --data_path "$jsonl_file"
    done


done
