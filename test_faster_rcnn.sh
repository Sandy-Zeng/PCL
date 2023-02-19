#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
ITER_PARAM=$4
OUTPUT_DIR=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  casia)
    TRAIN_IMDB="casia_train_all_single"
    TEST_IMDB="dist_test_all_single"
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  casia_unlabeled)
    TRAIN_IMDB="casia_train_all_single_semi_unlabeled_9"
    TEST_IMDB="casia_test_all_single"
    STEPSIZE=[40000,90000]
    ITERS=80000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  columbia)
    TRAIN_IMDB="casia_train_all_single"
    # TRAIN_IMDB="coco_train_filter_single"
    TEST_IMDB="dist_test_all_single"
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  dist_fake)
    TRAIN_IMDB="dist_cover_train_single"
    TEST_IMDB="dist_cover_test_single"
    ITERS=25000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  NIST)
    TRAIN_IMDB="dist_NIST_train_new_2"
    TEST_IMDB="dist_NIST_test_new_2"
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  NIST_unlabeled)
    TRAIN_IMDB="dist_NIST_train_new_2_semi"
    TEST_IMDB="dist_NIST_test_new_2"
    STEPSIZE=[30000,50000]
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_train_filter_single"
    TEST_IMDB="coco_test_filter_single"
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  imd2020)
    TRAIN_IMDB="imd2020_train_split_tamper"
    TEST_IMDB="imd2020_test_split_tamper"
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
#
LOG="./logs/test_${NET}_${ITER_PARAM}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${OUTPUT_DIR}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITER_PARAM}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITER_PARAM}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg ./cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg ./cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
fi
