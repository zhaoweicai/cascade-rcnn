// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/scale_route_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
    
template <typename Dtype>
void ScaleRouteLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // propagate_down=false for all bottoms
  if (this->layer_param_.propagate_down_size() == 0) {
    for (int k = 0; k < bottom.size(); k++) {
      this->layer_param_.add_propagate_down(false);
    }
  } else {
    for (int k = 0; k < bottom.size(); k++) {
      CHECK(!this->layer_param_.propagate_down(k));
    }
  }
  CHECK_EQ(bottom.size(),top.size());
}

template <typename Dtype>
void ScaleRouteLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num_rois = bottom[0]->num();
  // bottom[0]: rois (img_id, x1, y1, x2, y2)
  CHECK_EQ(bottom[0]->channels(),5);
  for (int k = 0; k < top.size(); k++) {
    CHECK_EQ(num_rois,bottom[k]->num());
    // dumming reshape
    const int channels = bottom[k]->channels();
    const int height = bottom[k]->height();
    const int width = bottom[k]->width();
    top[k]->Reshape(1, channels, height, width);
  }
}

template <typename Dtype>
void ScaleRouteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ScaleRouteParameter scale_route_param = this->layer_param_.scale_route_param();
  float low_scale, high_scale, mid_scale;
  if (scale_route_param.has_low_scale()) {
    low_scale = scale_route_param.low_scale();
    CHECK_GT(low_scale,0);
  } else {
    low_scale = 0.;
  }
  if (scale_route_param.has_high_scale()) {
    high_scale = scale_route_param.high_scale();
    CHECK_LT(high_scale,FLT_MAX);
  } else {
    high_scale = FLT_MAX;
  }
  if (scale_route_param.has_low_scale() && scale_route_param.has_high_scale()) {
    mid_scale = sqrt(low_scale*high_scale);
  } else if (scale_route_param.has_low_scale()) {
    mid_scale = low_scale;
  } else if (scale_route_param.has_high_scale()) {
    mid_scale = high_scale;
  } else {
    mid_scale = 256.;
  }
  CHECK_GE(high_scale,low_scale); CHECK_GT(mid_scale,0);
  
  const int num_rois = bottom[0]->num();
  const Dtype* rois_data = bottom[0]->cpu_data();
  const int rois_dim = bottom[0]->channels();
  vector<int> select_index;
  float min_mid_dist = FLT_MAX; int min_mid_idx = -1;
  for (int i = 0; i < num_rois; i++) {
    BBox bb = InitBBox(rois_data[i*rois_dim+1],rois_data[i*rois_dim+2],
            rois_data[i*rois_dim+3],rois_data[i*rois_dim+4]);
    float bb_size = BBoxSize(bb);
    float bb_scale = sqrt(bb_size);
    if (bb_scale>=low_scale && bb_scale<high_scale) {
      select_index.push_back(i);
    }
    float mid_dist = std::abs(log2(bb_scale/mid_scale));
    if (mid_dist<min_mid_dist) {
      min_mid_dist = mid_dist;
      min_mid_idx = i;
    }
  }
  // in case of no selected index
  if (select_index.size() == 0) {
    DLOG(INFO) <<"layer: "<<this->layer_param().name()<<", No samples between : "
            <<low_scale<<" and "<<high_scale<<"!!!";
    CHECK_GE(min_mid_idx,0);
    select_index.push_back(min_mid_idx);
  }
 
  // copy bottoms to tops
  const int select_num = select_index.size();
  for (int k = 0; k < top.size(); k++) {
    const int channels = bottom[k]->channels();
    const int height = bottom[k]->height();
    const int width = bottom[k]->width();
    const int dim = bottom[k]->count() / bottom[k]->num();   
    top[k]->Reshape(select_num, channels, height, width);
    
    const Dtype* bottom_data = bottom[k]->cpu_data();
    Dtype* top_data = top[k]->mutable_cpu_data();
    for (int i = 0; i < select_num; i++) {
      int idx = select_index[i]; CHECK_GT(bottom[k]->num(),idx);
      caffe_copy(dim, bottom_data+idx*dim, top_data+i*dim);
    }
  }
}

INSTANTIATE_CLASS(ScaleRouteLayer);
REGISTER_LAYER_CLASS(ScaleRoute);

}  // namespace caffe
