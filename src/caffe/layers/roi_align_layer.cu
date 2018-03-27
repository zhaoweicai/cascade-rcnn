// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype pad_ratio, const Dtype* bottom_rois, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    const int grid_width = pooled_width + 1;
    const int grid_height = pooled_height + 1;
    int pw = index % grid_width;
    int ph = (index / grid_width) % grid_height;
    int c = (index / grid_width / grid_height) % channels;
    int n = index / grid_width / grid_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    // padding
    Dtype pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio;
    Dtype pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio;

    Dtype roi_start_w = (bottom_rois[1]-pad_w) * spatial_scale;
    Dtype roi_start_h = (bottom_rois[2]-pad_h) * spatial_scale;
    Dtype roi_end_w = (bottom_rois[3]+pad_w) * spatial_scale;
    Dtype roi_end_h = (bottom_rois[4]+pad_h) * spatial_scale;

    // coordinate shift
    roi_start_w -= 0.5; roi_start_h -= 0.5;
    roi_end_w -= 0.5; roi_end_h -= 0.5;

    const Dtype roi_height = roi_end_h-roi_start_h;
    const Dtype roi_width = roi_end_w-roi_start_w;

    // set zero for malformed ROIs
    if (roi_height <= 0 || roi_width <= 0) {
      top_data[index] = Dtype(0);
      return;
    }

    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hfloat = roi_start_h + static_cast<Dtype>(ph)*bin_size_h;
    Dtype wfloat = roi_start_w + static_cast<Dtype>(pw)*bin_size_w;

    // set zero when grid is out of feature map
    if (hfloat < -0.5 || hfloat > (height-0.5) || 
            wfloat < -0.5 || wfloat > (width-0.5)) {
      top_data[index] = Dtype(0);
      return;
    }

    // neighboring feature coordinates
    int hfloor = floor(hfloat), wfloor = floor(wfloat);
    int hceil = hfloor+1, wceil = wfloor+1;

    // clipping
    hfloat = min(max(hfloat, Dtype(0)), static_cast<Dtype>(height-1));
    wfloat = min(max(wfloat, Dtype(0)), static_cast<Dtype>(width-1));
    hfloor = min(max(hfloor, 0), (height-1));
    wfloor = min(max(wfloor, 0), (width-1));
    hceil = min(max(hceil, 0), (height-1));
    wceil = min(max(wceil, 0), (width-1));

    // coefficients and features for bilinear interpolation
    Dtype lh = hfloat-hfloor, lw = wfloat-wfloor;
    Dtype hh = 1-lh, hw = 1-lw;
    Dtype w00 = hw*hh, w10 = lw*hh, w01 = hw*lh, w11 = lw*lh;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype v00 = bottom_data[hfloor*width + wfloor];
    Dtype v10 = bottom_data[hfloor*width + wceil];
    Dtype v01 = bottom_data[hceil*width + wfloor];
    Dtype v11 = bottom_data[hceil*width + wceil];

    Dtype val = w00*v00 + w10*v10 + w01*v01 + w11*v11;
    top_data[index] = val;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, pad_ratio_, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype pad_ratio, const Dtype* bottom_rois, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    const int grid_width = pooled_width + 1;
    const int grid_height = pooled_height + 1;
    int pw = index % grid_width;
    int ph = (index / grid_width) % grid_height;
    int c = (index / grid_width / grid_height) % channels;
    int n = index / grid_width / grid_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    // padding
    Dtype pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio;
    Dtype pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio;

    Dtype roi_start_w = (bottom_rois[1]-pad_w) * spatial_scale;
    Dtype roi_start_h = (bottom_rois[2]-pad_h) * spatial_scale;
    Dtype roi_end_w = (bottom_rois[3]+pad_w) * spatial_scale;
    Dtype roi_end_h = (bottom_rois[4]+pad_h) * spatial_scale;

    // coordinate shift
    roi_start_w -= 0.5; roi_start_h -= 0.5;
    roi_end_w -= 0.5; roi_end_h -= 0.5;

    const Dtype roi_height = roi_end_h-roi_start_h;
    const Dtype roi_width = roi_end_w-roi_start_w;

    // set zero for malformed ROIs
    if (roi_height <= 0 || roi_width <= 0) {
      return;
    }

    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hfloat = roi_start_h + static_cast<Dtype>(ph)*bin_size_h;
    Dtype wfloat = roi_start_w + static_cast<Dtype>(pw)*bin_size_w;

    // set zero when grid is out of feature map
    if (hfloat < -0.5 || hfloat > (height-0.5) || 
            wfloat < -0.5 || wfloat > (width-0.5)) {
      return;
    }

    // neighboring feature coordinates
    int hfloor = floor(hfloat), wfloor = floor(wfloat);
    int hceil = hfloor+1, wceil = wfloor+1;

    // clipping
    hfloat = min(max(hfloat, Dtype(0)), static_cast<Dtype>(height-1));
    wfloat = min(max(wfloat, Dtype(0)), static_cast<Dtype>(width-1));
    hfloor = min(max(hfloor, 0), (height-1));
    wfloor = min(max(wfloor, 0), (width-1));
    hceil = min(max(hceil, 0), (height-1));
    wceil = min(max(wceil, 0), (width-1));

    // coefficients and features for bilinear interpolation
    Dtype lh = hfloat-hfloor, lw = wfloat-wfloor;
    Dtype hh = 1-lh, hw = 1-lw;
    Dtype w00 = hw*hh, w10 = lw*hh, w01 = hw*lh, w11 = lw*lh;

    const int base_index = (roi_batch_ind * channels + c) * height * width;
    const int v00_index = base_index + hfloor*width + wfloor;
    const int v10_index = base_index + hfloor*width + wceil;
    const int v01_index = base_index + hceil*width + wfloor;
    const int v11_index = base_index + hceil*width + wceil;
    const Dtype v00_gradient = w00*top_diff[index];
    const Dtype v10_gradient = w10*top_diff[index];
    const Dtype v01_gradient = w01*top_diff[index];
    const Dtype v11_gradient = w11*top_diff[index];
    caffe_gpu_atomic_add(v00_gradient, bottom_diff + v00_index);
    caffe_gpu_atomic_add(v10_gradient, bottom_diff + v10_index);
    caffe_gpu_atomic_add(v01_gradient, bottom_diff + v01_index);
    caffe_gpu_atomic_add(v11_gradient, bottom_diff + v11_index);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = top[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, pad_ratio_, bottom_rois, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
