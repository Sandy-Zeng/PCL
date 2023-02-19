#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
FC=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  casia)
    TRAIN_IMDB="casia_train_all_single"
    TEST_IMDB="casia_test_all_single"
    STEPSIZE=[40000,60000]
    ITERS=80000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  dist_fake)
    TRAIN_IMDB="dist_cover_train_single"
    TEST_IMDB="dist_cover_test_single"
    STEPSIZE=10000
    ITERS=25000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  NIST)
    TRAIN_IMDB="dist_NIST_train_new_2"
    TEST_IMDB="dist_NIST_test_new_2"
    STEPSIZE=[30000,40000]
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  NIST_unlabeled)
    TRAIN_IMDB="dist_NIST_train_new_2_semi"
    TEST_IMDB="dist_NIST_test_new_2"
    STEPSIZE=[30000,40000]
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_train_filter_single"
    TEST_IMDB="coco_test_filter_single"
    STEPSIZE=[80000,160000]
    ITERS=130000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  imd2020)
    TRAIN_IMDB="imd2020_train_split_tamper"
    TEST_IMDB="imd2020_test_split_tamper"
    STEPSIZE=[20000,30000]
    ITERS=40000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="./logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

WEIGHT_PATH='./data/imagenet_weights/resnet_v1_101.ckpt'
if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./tools/trainval_net.py \
            --weight ${WEIGHT_PATH} \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --loadfc ${FC} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS} 
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./tools/trainval_net.py \
            --weight ${WEIGHT_PATH} \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --net ${NET} \
            --loadfc ${FC} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS} 
    fi
fi
