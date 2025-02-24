#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/bash/output
#model=qwen2-vl-7b-instruct
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct
model=internvl2-2b
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-2B
round=10
val_dataset=/ailab/user/wangwenhao/ms-swift/output/gt_val_200_v1.json
peft_list=(
# high train
#v2-20241226-201832
#v5-20241226-205945
#v12-20241228-154530
#v13-20241229-070835
#v14-20241229-123748
#v15-20241230-051644
#v16-20241231-173016
#v17-20241231-173046
#v20-20241231-174433
#v23-20241231-174820
#v24-20250101-143901
# low train
#v18-20241231-173219
#v19-20241231-173829
#v21-20241231-174712
#v22-20241231-174747
#v25-20250101-143901
#v26-20250101-144216
#v27-20250102-190101
#v28-20250103-001710
#v50-20250106-010846
#v51-20250106-010951
#v52-20250106-011321
#v53-20250106-011548
#v55-20250106-012449
#v57-20250106-165042

#v31-20250103-062506
#v32-20250103-090216
#v33-20250103-114428
#v34-20250103-142412
#v35-20250103-142412
#v36-20250103-170603
#v37-20250103-170603
#v38-20250103-194451
v39-20250105-000001
#v40-20250105-000030

#v41-20250105-000132
#v42-20250105-172655
#v43-20250105-172735
#v44-20250105-172920
#v45-20250105-173844
#v46-20250105-174028
#v47-20250106-010547
v48-20250106-010625
#v49-20250106-010728




#v59-20250117-161332
#v60-20250117-194907
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
