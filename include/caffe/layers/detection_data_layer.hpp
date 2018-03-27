// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_DETECTION_DATA_LAYER_HPP_
#define CAFFE_DETECTION_DATA_LAYER_HPP_

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
class DetectionDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DetectionDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DetectionDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void ShuffleList();
  virtual void ShuffleAspectGroupList();
  virtual void ShuffleMixAspectGroupList();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  vector<vector<BBox> > windows_;
  vector<vector<BBox> > roni_windows_;
  vector<Dtype> mean_values_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
  vector<int> strides_;
  vector<int> field_ws_;
  vector<int> field_hs_;
  int label_channel_;
  int label_blob_num_;
  vector<int> image_list_;
  vector<int> longer_width_list_;
  vector<int> longer_height_list_;
  int mix_group_size_;
  int list_id_;
  bool output_gt_boxes_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_DATA_LAYER_HPP_
