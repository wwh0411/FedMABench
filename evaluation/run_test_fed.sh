#!/bin/bash

base_path=/ailab/user/wangwenhao/FedMobile/bash/output
model=internvl2-1b
model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-1B
#model=internvl2-8b
#model_id_or_path=/ailab/user/wangwenhao/.cache/modelscope/hub/OpenGVLab/InternVL2-8B
round=10
val_dataset=/ailab/user/wangwenhao/ms-swift/c5_n100_val_apphete.jsonl
#val_dataset=/ailab/user/wangwenhao/ms-swift/c5_n100_val2_random.jsonl
peft_list=(
v11-20250221-183012
#v61-20250221-183012
#v0-20250221-183012
#v111-20250111-192005
#v112-20250111-192118
#v113-20250111-192903
#v114-20250111-193503
#v115-20250112-013642
#v116-20250112-014155
#v117-20250112-014259
#v118-20250112-022607
#v119-20250112-022857
#v120-20250112-162008
#v121-20250112-164333
#v122-20250112-164548
#v123-20250112-164813
#v124-20250112-165712

#v131-20250112-192200
#v132-20250112-192531
#v133-20250112-231300
#v134-20250112-231733
#v135-20250112-231958
#v139-20250113-122609
#v142-20250114-012316
#v143-20250114-012437
#v144-20250114-012722
#v145-20250114-012922
#v147-20250114-023037
#v156-20250115-020003
#v157-20250117-204748
#v158-20250117-204832
#v161-20250121-210352
#v162-20250121-232520
#v163-20250121-233547
#v164-20250121-233708

#v199-20250208-002248
#v200-20250208-002803
#v201-20250208-003735
#v202-20250208-003820
#v203-20250208-003919
#v204-20250208-003950
#v205-20250208-172014
#v206-20250208-172221
#v207-20250208-184430
#v208-20250208-185847
#v209-20250208-203103
#v210-20250208-204357
#v211-20250208-210254
#v212-20250208-210430
#v213-20250208-210558
#v214-20250208-213912
#v215-20250208-214008
#v216-20250208-214050
#v217-20250209-015313
#v218-20250209-015331
#v219-20250209-015353
#v220-20250209-015413
#v221-20250209-015502
#v222-20250209-015529
#v223-20250209-015640
#v224-20250209-015751
#v225-20250209-020232
#v226-20250209-020305
#v227-20250209-121548
#v228-20250209-121752
#v229-20250209-121846
#v230-20250209-123121
#v231-20250209-123157
#v232-20250209-123228
#v233-20250209-123258


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
    python test_swift_app.py --data_path "$jsonl_file" --category_file /ailab/user/wangwenhao/ms-swift/val_app_hete.json
    done


done
