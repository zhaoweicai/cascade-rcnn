
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights=../../../models/resnet/ResNet-101-model-merge.caffemodel \
  --gpu=0,1  2>&1 | tee log.txt