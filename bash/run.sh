#!/bin/bash
# script1.sh

#bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg4_train_5000_v1.json $1
#bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg5_train_5000_v1.json $1
#bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/gt_train_5000_v1.json $1
#bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg3_train_5000_v1.json $1


#bash run_central_internvl2.sh /ailab/user/wangwenhao/ms-swift/output/high/gt_train_5000_v1.json $1
#bash run_central_internvl2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg3_train_5000_v1.json $1
bash run_central_internvl2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg4_train_5000_v1.json $1
bash run_central_internvl2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg5_train_5000_v1.json $1

bash run_central_internvl2_3.sh /ailab/user/wangwenhao/ms-swift/output/high/alg4_train_5000_v1.json $1
#bash run_central_internvl2_3.sh /ailab/user/wangwenhao/ms-swift/output/high/alg5_train_5000_v1.json $1
bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/gt_train_5000_v1.json $1
#bash run_central_internvl2_2.sh /ailab/user/wangwenhao/ms-swift/output/high/alg3_train_5000_v1.json $1