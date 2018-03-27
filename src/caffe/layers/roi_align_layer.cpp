// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  grid_height_ = pooled_height_+1;
  grid_width_ = pooled_width_+1;
  spatial_scale_ = roi_pool_param.spatial_scale();
  pad_ratio_ = roi_pool_param.pad_ratio();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, grid_height_, grid_width_);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    
    // padding
    Dtype pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio_;
    Dtype pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio_;
    
    // start and end float coordinates at feature map scale
    Dtype roi_start_w = (bottom_rois[1]-pad_w) * spatial_scale_;
    Dtype roi_start_h = (bottom_rois[2]-pad_h) * spatial_scale_;
    Dtype roi_end_w = (bottom_rois[3]+pad_w) * spatial_scale_;
    Dtype roi_end_h = (bottom_rois[4]+pad_h) * spatial_scale_;
    
    // coordinate shift
    roi_start_w -= 0.5; roi_start_h -= 0.5;
    roi_end_w -= 0.5; roi_end_h -= 0.5;
       
    const Dtype roi_height = roi_end_h-roi_start_h;
    const Dtype roi_width = roi_end_w-roi_start_w;
    
    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph <= pooled_height_; ++ph) {
        for (int pw = 0; pw <= pooled_width_; ++pw) {
          const int pool_index = ph * grid_width_ + pw;
          // set zero for malformed ROIs
          if (roi_height <= 0 || roi_width <= 0) {
            top_data[pool_index] = Dtype(0);
            continue;
          }
          
          // float grid coordinates
          Dtype hfloat = roi_start_h + static_cast<Dtype>(ph)*bin_size_h;
          Dtype wfloat = roi_start_w + static_cast<Dtype>(pw)*bin_size_w;
                    
          // set zero when grid is out of feature map
          if (hfloat < -0.5 || hfloat > (height_-0.5) || 
                  wfloat < -0.5 || wfloat > (width_-0.5)) {
            top_data[pool_index] = Dtype(0);
            continue;
          }
          
          // neighboring feature coordinates
          int hfloor = floor(hfloat), wfloor = floor(wfloat);
          int hceil = hfloor+1, wceil = wfloor+1;
          
          // clipping
          hfloat = min(max(hfloat, Dtype(0)), static_cast<Dtype>(height_-1));
          wfloat = min(max(wfloat, Dtype(0)), static_cast<Dtype>(width_-1));
          hfloor = min(max(hfloor, 0), (height_-1));
          wfloor = min(max(wfloor, 0), (width_-1));
          hceil = min(max(hceil, 0), (height_-1));
          wceil = min(max(wceil, 0), (width_-1));

          // coefficients and features for bilinear interpolation
          Dtype lh = hfloat-hfloor, lw = wfloat-wfloor;
          Dtype hh = 1-lh, hw = 1-lw;
          CHECK_GE(lh,0); CHECK_LE(lh,1);
          CHECK_GE(lw,0); CHECK_LE(lw,1);
          CHECK_GE(hh,0); CHECK_LE(hh,1);
          CHECK_GE(hw,0); CHECK_LE(hw,1);
          Dtype w00 = hw*hh, w10 = lw*hh, w01 = hw*lh, w11 = lw*lh;
          
          Dtype v00 = batch_data[hfloor*width_+wfloor];
          Dtype v10 = batch_data[hfloor*width_+wceil];
          Dtype v01 = batch_data[hceil*width_+wfloor];
          Dtype v11 = batch_data[hceil*width_+wceil];
          
          // bilinear interpolation
          Dtype val = w00*v00 + w10*v10 + w01*v01 + w11*v11;
          top_data[pool_index] = val;
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
