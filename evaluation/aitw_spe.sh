#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/bash/output
model=qwen2-vl-7b-instruct
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
#model=internvl2-8b
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-8B
round_list=(5 15 20 25 30)  # 修改为一个列表
val_dataset=/ailab/user/wangwenhao/ms-swift/output_aitw
general_list=(
# aitw
#v74-20250107-231433
#v75-20250108-162852
#v38-20250103-004324
#v39-20250103-004622
#v76-20250108-165445
#v77-20250108-170133
#v78-20250108-170615
#v81-20250109-023328
#v177-20250128-170428
v169-20250125-225727
)
google_list=(
#v80-20250108-202058
#v82-20250109-023419
#v84-20250109-024640
#v90-20250109-152853
#v96-20250109-230607
v178-20250128-191652
)
install_list=(
#v79-20250108-183816
#v85-20250109-025523
#v86-20250109-115343
#v89-20250109-152834
#v93-20250109-183516
v179-20250128-203901
)
web_list=(
#v87-20250109-115449
#v88-20250109-115542
#v91-20250109-152923
#v92-20250109-155048
#v95-20250109-230544
v181-20250128-231241
)
single_list=(
#v98-20250110-115714
#v99-20250110-115915
##v100-20250110-132400
#v101-20250110-134946
##v102-20250110-152758
##v103-20250110-153647
#v104-20250110-195351
#v105-20250110-200914
v180-20250128-222709
)
# general
for round in ${round_list[@]}; do  # 遍历每个 round

    for i in ${general_list[@]};
    do
        echo $i

        # inference
        # check already exsist
        jsonl_files=$(find "$base_path/$model/$i/global_lora_$round/infer_result" -type f -name "*.jsonl")

        if [ -z "$jsonl_files" ]; then
        MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=$1 swift infer --ckpt_dir "$base_path/$model/$i/global_lora_$round" \
          --val_dataset "$val_dataset/general_gt_val_100_v1.json" --model_type $model --model_id_or_path $model_id_or_path --sft_type lora
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
done