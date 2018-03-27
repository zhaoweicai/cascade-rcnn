
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights=../../../models/vggnet/VGG_16_fc2048_prune.caffemodel \
  --gpu=0,1  2>&1 | tee log.txt