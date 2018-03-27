// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include "thrust/functional.h"
#include "thrust/sort.h"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EncodeBBoxesKernel(const int nthreads, const Dtype* bbox_data, 
        const Dtype* label, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const float fw, const float fh, const float stride, 
        Dtype* encode_data) {
  const int spatial_dim = width*height;
  const int bbox_dim = 4*spatial_dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim;
    const int s = index % spatial_dim;
    const int idx = i*bbox_dim + s; 
    const int label_value = static_cast<int>(label[index]);
    if (label_value > 0) {
      const int h = s / width, w = s % width; 
      Dtype bbx, bby, bbw, bbh; 
      bbx = (bbox_data[idx]-(w+Dtype(0.5))*stride) / fw;
      bby = (bbox_data[idx+spatial_dim]-(h+Dtype(0.5))*stride) / fh;
      bbw = log(max(bbox_data[idx+2*spatial_dim],Dtype(2)) / fw);
      bbh = log(max(bbox_data[idx+3*spatial_dim],Dtype(2)) / fh);   
      // bbox normalization
      bbx -= means[0]; bby -= means[1]; bbw -= means[2]; bbh -= means[3];
      bbx /= stds[0]; bby /= stds[1]; bbw /= stds[2]; bbh /= stds[3];
     
      encode_data[idx] = bbx;
      encode_data[idx+spatial_dim] = bby;
      encode_data[idx+2*spatial_dim] = bbw;
      encode_data[idx+3*spatial_dim] = bbh;
    } else {
      encode_data[idx] = Dtype(0);
      encode_data[idx+spatial_dim] = Dtype(0);
      encode_data[idx+2*spatial_dim] = Dtype(0);
      encode_data[idx+3*spatial_dim] = Dtype(0);
    }
  }
}

template <typename Dtype>
void EncodeBBoxesGPU(const int nthreads, const Dtype* bbox_data, const Dtype* label,
        const int width, const int height, const Dtype* means, const Dtype* stds, 
        const float fw, const float fh, const float stride, Dtype* encode_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  EncodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, label, width, height, means, 
      stds, fw, fh, stride, encode_data);
  CUDA_POST_KERNEL_CHECK;
}

template 
void EncodeBBoxesGPU(const int nthreads, const float* bbox_data, const float* label,
        const int width, const int height, const float* means, const float* stds, 
        const float fw, const float fh, const float stride, float* encode_data);

template 
void EncodeBBoxesGPU(const int nthreads, const double* bbox_data, const double* label,
        const int width, const int height, const double* means, const double* stds, 
        const float fw, const float fh, const float stride, double* encode_data);

template <typename Dtype>
__global__ void EncodeGroupBBoxesKernel(const int nthreads, const Dtype* bbox_data, 
        const Dtype* label, const int group_num, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* fws, const Dtype* fhs, 
        const float stride, Dtype* encode_data) {
  const int spatial_dim = width*height;
  const int bbox_dim = 4*spatial_dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim; // group index
    const int k = i % group_num; 
    const int s = index % spatial_dim;
    const int idx = i*bbox_dim + s; 
    const int label_value = static_cast<int>(label[index]);
    if (label_value > 0) {
      const int h = s / width, w = s % width; 
      Dtype bbx, bby, bbw, bbh; 
      bbx = (bbox_data[idx]-(w+Dtype(0.5))*stride) / fws[k];
      bby = (bbox_data[idx+spatial_dim]-(h+Dtype(0.5))*stride) / fhs[k];
      bbw = log(max(bbox_data[idx+2*spatial_dim],Dtype(2)) / fws[k]);
      bbh = log(max(bbox_data[idx+3*spatial_dim],Dtype(2)) / fhs[k]);   
      // bbox normalization
      bbx -= means[0]; bby -= means[1]; bbw -= means[2]; bbh -= means[3];
      bbx /= stds[0]; bby /= stds[1]; bbw /= stds[2]; bbh /= stds[3];
     
      encode_data[idx] = bbx;
      encode_data[idx+spatial_dim] = bby;
      encode_data[idx+2*spatial_dim] = bbw;
      encode_data[idx+3*spatial_dim] = bbh;
    } else {
      encode_data[idx] = Dtype(0);
      encode_data[idx+spatial_dim] = Dtype(0);
      encode_data[idx+2*spatial_dim] = Dtype(0);
      encode_data[idx+3*spatial_dim] = Dtype(0);
    }
  }
}

template <typename Dtype>
void EncodeGroupBBoxesGPU(const int nthreads, const Dtype* bbox_data, 
        const Dtype* label, const int group_num, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* fws, const Dtype* fhs, 
        const float stride, Dtype* encode_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  EncodeGroupBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, label, group_num, width, 
      height, means, stds, fws, fhs, stride, encode_data);
  CUDA_POST_KERNEL_CHECK;
}

template 
void EncodeGroupBBoxesGPU(const int nthreads, const float* bbox_data, 
        const float* label, const int group_num, const int width, const int height, 
        const float* means, const float* stds, const float* fws, const float* fhs, 
        const float stride, float* encode_data);

template 
void EncodeGroupBBoxesGPU(const int nthreads, const double* bbox_data, 
        const double* label, const int group_num, const int width, const int height, 
        const double* means, const double* stds, const double* fws, const double* fhs, 
        const float stride, double* encode_data);

template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads, const Dtype* bbox_data, 
        const int bottom_dim, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* bounds, const float fw, 
        const float fh, const float stride, const int top_dim, Dtype* pred_data) {
  const int spatial_dim = width*height;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim;
    const int s = index % spatial_dim;
    const int idx = i*bottom_dim + s;
    const int h = s / width, w = s % width; 

    Dtype bbx, bby, bbw, bbh;
    // bbox de-normalization
    bbx = bbox_data[idx]*stds[0]+means[0];
    bby = bbox_data[idx+spatial_dim]*stds[1]+means[1];
    bbw = bbox_data[idx+2*spatial_dim]*stds[2]+means[2];
    bbh = bbox_data[idx+3*spatial_dim]*stds[3]+means[3];
       
    // bbox bounding
    bbx = max(bounds[0],bbx); bbx = min(bounds[1],bbx); 
    bby = max(bounds[0],bby); bby = min(bounds[1],bby);
    bbw = max(bounds[2],bbw); bbw = min(bounds[3],bbw); 
    bbh = max(bounds[2],bbh); bbh = min(bounds[3],bbh);

    bbx = bbx*fw + (w+Dtype(0.5))*stride;
    bby = bby*fh + (h+Dtype(0.5))*stride;         
    bbw = fw*exp(bbw); bbh = fh*exp(bbh);
    bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
        
    const int top_idx = i*top_dim+s;
    pred_data[top_idx] = bbx; pred_data[top_idx+spatial_dim] = bby; 
    pred_data[top_idx+2*spatial_dim] = bbx+bbw-1; 
    pred_data[top_idx+3*spatial_dim] = bby+bbh-1; 
  }
}

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads, const Dtype* bbox_data, 
        const int bottom_dim, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, Dtype* pred_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, bottom_dim, width, 
      height, means, stds, bounds, fw, fh, stride, top_dim, pred_data);
  CUDA_POST_KERNEL_CHECK;
}

template 
void DecodeBBoxesGPU(const int nthreads, const float* bbox_data, 
        const int bottom_dim, const int width, const int height, const float* means, 
        const float* stds, const float* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, float* pred_data);

template 
void DecodeBBoxesGPU(const int nthreads, const double* bbox_data, 
        const int bottom_dim, const int width, const int height, const double* means, 
        const double* stds, const double* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, double* pred_data);

template <typename Dtype>
__global__ void DecodeGroupBBoxesKernel(const int nthreads, const Dtype* bbox_data, 
        const int group_num, const int bottom_dim, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* bounds, const Dtype* fws, 
        const Dtype* fhs, const float stride, const int top_dim, Dtype* pred_data) {
  const int spatial_dim = width*height;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim;
    const int k = i % group_num; 
    const int s = index % spatial_dim;
    const int idx = i*bottom_dim + s;
    const int h = s / width, w = s % width; 

    Dtype bbx, bby, bbw, bbh;
    // bbox de-normalization
    bbx = bbox_data[idx]*stds[0]+means[0];
    bby = bbox_data[idx+spatial_dim]*stds[1]+means[1];
    bbw = bbox_data[idx+2*spatial_dim]*stds[2]+means[2];
    bbh = bbox_data[idx+3*spatial_dim]*stds[3]+means[3];
       
    // bbox bounding
    bbx = max(bounds[0],bbx); bbx = min(bounds[1],bbx); 
    bby = max(bounds[0],bby); bby = min(bounds[1],bby);
    bbw = max(bounds[2],bbw); bbw = min(bounds[3],bbw); 
    bbh = max(bounds[2],bbh); bbh = min(bounds[3],bbh);

    bbx = bbx*fws[k] + (w+Dtype(0.5))*stride;
    bby = bby*fhs[k] + (h+Dtype(0.5))*stride;         
    bbw = fws[k]*exp(bbw); bbh = fhs[k]*exp(bbh);
    bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
        
    const int top_idx = i*top_dim+s;
    pred_data[top_idx] = bbx; pred_data[top_idx+spatial_dim] = bby; 
    pred_data[top_idx+2*spatial_dim] = bbx+bbw-1; 
    pred_data[top_idx+3*spatial_dim] = bby+bbh-1; 
  }
}

template <typename Dtype>
void DecodeGroupBBoxesGPU(const int nthreads, const Dtype* bbox_data, 
        const int group_num, const int bottom_dim, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* bounds, const Dtype* fws, 
        const Dtype* fhs, const float stride, const int top_dim, Dtype* pred_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodeGroupBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, group_num, bottom_dim, width, 
      height, means, stds, bounds, fws, fhs, stride, top_dim, pred_data);
  CUDA_POST_KERNEL_CHECK;
}

template 
void DecodeGroupBBoxesGPU(const int nthreads, const float* bbox_data, 
        const int group_num, const int bottom_dim, const int width, const int height, 
        const float* means, const float* stds, const float* bounds, const float* fws, 
        const float* fhs, const float stride, const int top_dim, float* pred_data);

template 
void DecodeGroupBBoxesGPU(const int nthreads, const double* bbox_data, 
        const int group_num, const int bottom_dim, const int width, const int height, 
        const double* means, const double* stds, const double* bounds, const double* fws, 
        const double* fhs, const float stride, const int top_dim, double* pred_data);

template <typename Dtype>
__global__ void BoundBBoxPredsKernel(const int nthreads, Dtype* bbox_data, 
        const int bottom_dim, const int sp_dim, const Dtype* bounds) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / sp_dim;
    const int s = index % sp_dim;
    const int idx = i*bottom_dim+s;
    // x
    bbox_data[idx] = max(bounds[0],bbox_data[idx]);
    bbox_data[idx] = min(bounds[1],bbox_data[idx]);
    // y
    bbox_data[idx+sp_dim] = max(bounds[0],bbox_data[idx+sp_dim]);
    bbox_data[idx+sp_dim] = min(bounds[1],bbox_data[idx+sp_dim]);
    // w
    bbox_data[idx+2*sp_dim] = max(bounds[2],bbox_data[idx+2*sp_dim]);
    bbox_data[idx+2*sp_dim] = min(bounds[3],bbox_data[idx+2*sp_dim]);
    // h
    bbox_data[idx+3*sp_dim] = max(bounds[2],bbox_data[idx+3*sp_dim]);
    bbox_data[idx+3*sp_dim] = min(bounds[3],bbox_data[idx+3*sp_dim]);
  }
}

template <typename Dtype>
void BoundBBoxPredsGPU(const int nthreads, Dtype* bbox_data, 
        const int bottom_dim, const int sp_dim, const Dtype* bounds) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  BoundBBoxPredsKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, bottom_dim, sp_dim, bounds);
  CUDA_POST_KERNEL_CHECK;
}

template 
void BoundBBoxPredsGPU(const int nthreads, float* bbox_data, 
        const int bottom_dim, const int sp_dim, const float* bounds);

template 
void BoundBBoxPredsGPU(const int nthreads, double* bbox_data, 
        const int bottom_dim, const int sp_dim, const double* bounds);

}  // namespace caffe
