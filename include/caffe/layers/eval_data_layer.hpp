// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_EVAL_DATA_LAYER_HPP_
#define CAFFE_EVAL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
class EvalDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit EvalDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~EvalDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EvalData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, vector<int> > > image_database_;
  vector<vector<BBox> > windows_;
  vector<vector<BBox> > roni_windows_;
  vector<Dtype> mean_values_;
  bool has_mean_values_;
  vector<int> image_list_;
  int list_id_;
  bool output_image_size_;
};

}  // namespace caffe

#endif  // CAFFE_EVAL_DATA_LAYER_HPP_
