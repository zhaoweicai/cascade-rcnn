
GLOG_logtostderr=1 ../../../build/tools/caffe testDetection \
  --model=test.prototxt \
  --weights=cascadercnn_coco_iter_280000.caffemodel \
  --gpu=0 \
  --ap_version=101point \
  --iterations=5000  2>&1 | tee log_test_280k_det.txt
