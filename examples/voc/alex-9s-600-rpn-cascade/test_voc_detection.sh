
GLOG_logtostderr=1 ../../../build/tools/caffe testDetection \
  --model=test.prototxt \
  --weights=cascadercnn_voc_iter_45000.caffemodel \
  --gpu=0 \
  --ap_version=101point \
  --iterations=4952  2>&1 | tee log_test_45k_det.txt
