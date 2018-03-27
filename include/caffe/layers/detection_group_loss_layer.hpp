// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_DETECTION_GROUP_LOSS_LAYERS_HPP_
#define CAFFE_DETECTION_GROUP_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DetectionGroupLossLayer : public LossLayer<Dtype> {
 public:
  explicit DetectionGroupLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionGroupLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  
  Blob<Dtype> gt_bbox_;
  Blob<Dtype> gt_label_;
  Blob<Dtype> gt_overlap_;
  Blob<Dtype> bbox_diff_;
  Blob<Dtype> keep_map_;
  Blob<Dtype> weight_map_;
  
  shared_ptr<Caffe::RNG> shuffle_rng_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  int group_num_;
  int cls_num_;
  int coord_num_;
  int batch_size_;
  float lambda_;
  float stride_;
  float bg_threshold_;
  float bg_multiple_;
  string neg_ranking_type_;
  string neg_mining_type_;
  int  min_num_neg_;
  bool objectness_;
  bool pos_neg_weighted_;
  bool do_bound_bbox_;
  string bbox_loss_type_;
  
  Blob<Dtype> field_ws_;
  Blob<Dtype> field_hs_;
  Blob<Dtype> bbox_mean_;
  Blob<Dtype> bbox_std_;
  Blob<Dtype> bbox_bound_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_GROUP_LOSS_LAYERS_HPP_
