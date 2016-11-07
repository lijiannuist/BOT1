#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/data2/lijian/BOT
DATA=/data2/lijian/BOT/code
TOOLS=/data2/lijian/caffe-master/build/tools

$TOOLS/compute_image_mean $EXAMPLE/animal_train_lmdb \
  $DATA/mean300.binaryproto

echo "Done."
