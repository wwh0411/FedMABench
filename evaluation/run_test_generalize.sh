#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/bash/output
model=qwen2-vl-7b-instruct
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
#model=internvl2-8b
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-8B
round=10
val_dataset=(
#/ailab/user/wangwenhao/ms-swift/output_aitw/general_gt_val_100_v1.json
/ailab/user/wangwenhao/ms-swift/output/generalize/iid_gt_high_val_60.json
/ailab/user/wangwenhao/ms-swift/output/generalize/task_gt_high_val_60.json
/ailab/user/wangwenhao/ms-swift/output/generalize/app_gt_high_val_60.json
/ailab/user/wangwenhao/ms-swift/output/generalize/cate_gt_high_val_60.json
)
peft_list=(
# high
#v110-20250111-191915
#v111-20250111-192005
##v112-20250111-192118
##v113-20250111-192903
#v114-20250111-193503
v123-20250112-164813
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
    jsonl_files=$(find "$base_path/$model/$i/global_lora_$round/infer_result" -type f -name "*.json")

    if [ -z "$jsonl_files" ]; then
    for val in ${val_dataset[@]};
    do
      MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=$1 swift infer --ckpt_dir "$base_path/$model/$i/global_lora_$round" \
        --val_dataset $val --model_type $model --model_id_or_path $model_id_or_path --sft_type lora
    done
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
