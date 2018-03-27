
GLOG_logtostderr=1 ../../../build/tools/caffe testDetection \
  --model=test.prototxt \
  --weights=cascadercnn_coco_iter_180000.caffemodel \
  --gpu=0 \
  --ap_version=101point \
  --iterations=5000  2>&1 | tee log_test_180k_det.txt
