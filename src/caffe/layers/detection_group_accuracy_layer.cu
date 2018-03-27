// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/detection_group_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DetGroupAccForwardGPU(const int nthreads,
          const Dtype* conf_data, const Dtype* gt_label, const Dtype* gt_overlap, 
          const int spatial_dim, const int conf_dim, const int cls_num,
          const Dtype bg_thr, const bool objectness, const bool has_ignore_label_, 
          const int ignore_label_, Dtype* counts, Dtype* fore_counts, Dtype* acc, 
          Dtype* fore_acc) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(gt_label[index]);
    if (objectness) {
      label_value = min(1,label_value);
    }
    if (has_ignore_label_ && label_value == ignore_label_) {
      acc[index] = 0; fore_acc[index] = 0;
      counts[index] = 0; fore_counts[index] = 0;
    } else if (label_value == 0 && gt_overlap[index] >= bg_thr) {
      acc[index] = 0; fore_acc[index] = 0;
      counts[index] = 0; fore_counts[index] = 0;
    } else {
      counts[index] = 1;
      if (label_value != 0) {
        fore_counts[index] = 1;
      }
      Dtype max_score = -FLT_MAX; int max_id = -1;
      for (int k = 0; k < cls_num; k++) {
        if (conf_data[n*conf_dim+k*spatial_dim+s] > max_score) {
          max_score = conf_data[n*conf_dim+k*spatial_dim+s];
          max_id = k;
        }
      }
      if (max_id == label_value) {
        acc[index] = 1;
        if (label_value != 0) {
          fore_acc[index] = 1; 
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void DetGroupBboxIouForwardGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype* gt_label, const Dtype* gt_bbox, 
          const int group_num, const int width, const int height, 
          const Dtype* bbox_mean, const Dtype* bbox_std, const Dtype* field_ws, 
          const Dtype* field_hs, const Dtype stride, const Dtype field_whr, 
          const Dtype field_xyr, const bool has_ignore_label_, const int ignore_label_, 
          Dtype* counts, Dtype* bbox_ious) {
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;
  const int spatial_dim = height*width;
  const int bbox_dim = 4*spatial_dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int k = n % group_num; 
    const int s = index % spatial_dim;
    const int h = s / width, w = s % width; 
    const int label_value = static_cast<int>(gt_label[index]);
    if ((has_ignore_label_ && label_value == ignore_label_) || (label_value==0)) {
      counts[index] = 0; bbox_ious[index] = 0;
    } else {
      Dtype tx, ty, tw, th; 
      const int bbox_idx = n*bbox_dim + s;
      tx = bbox_data[bbox_idx];
      ty = bbox_data[bbox_idx+spatial_dim];
      tw = bbox_data[bbox_idx+2*spatial_dim];
      th = bbox_data[bbox_idx+3*spatial_dim];
        
      // bbox de-normalization
      tx *= bbox_std[0]; ty *= bbox_std[1];
      tw *= bbox_std[2]; th *= bbox_std[3];
      tx += bbox_mean[0]; ty += bbox_mean[1];
      tw += bbox_mean[2]; th += bbox_mean[3];

      // bbox bounding
      tx = max(min_xyr,tx); tx = min(max_xyr,tx); 
      ty = max(min_xyr,ty); ty = min(max_xyr,ty);
      tw = max(min_whr,tw); tw = min(max_whr,tw); 
      th = max(min_whr,th); th = min(max_whr,th);
        
      tx = tx*field_ws[k] + (w+Dtype(0.5))*stride;
      ty = ty*field_hs[k] + (h+Dtype(0.5))*stride;
      tw = field_ws[k] * exp(tw); th = field_hs[k] * exp(th);
      tx = tx - tw/Dtype(2); ty = ty - th/Dtype(2);

      Dtype gx, gy, gw, gh;
      gx = gt_bbox[bbox_idx]; 
      gy = gt_bbox[bbox_idx+spatial_dim];
      gw = gt_bbox[bbox_idx+2*spatial_dim]; 
      gh = gt_bbox[bbox_idx+3*spatial_dim];
      gx = gx - gw/Dtype(2); gy = gy - gh/Dtype(2);
      
      // iou
      Dtype iou;
      if (tw<=0 || th<=0 || gw<=0 || gh<=0) {
        iou = 0;
      } else {
        Dtype tlx = max(tx, gx); Dtype tly = max(ty, gy);
        Dtype brx = min(tx+tw-1, gx+gw-1);
        Dtype bry = min(ty+th-1, gy+gh-1);
        Dtype over;
        if((tlx>brx)||(tly>bry)) over = Dtype(0);
        else over = (brx-tlx+1)*(bry-tly+1);
        Dtype u = tw*th+gw*gh-over;
        iou = over/u;
      }
      bbox_ious[index] = iou;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void DetectionGroupAccuracyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* conf_data = bottom[0]->gpu_data();
  const Dtype* bbox_data = bottom[1]->gpu_data();
  const Dtype* bottom_gt_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num() / group_num_;
  const int height = bottom[0]->height(); 
  const int width = bottom[0]->width();
  const int spatial_dim = height*width;
  const int conf_dim = bottom[0]->count() / bottom[0]->num();
  const int bbox_dim = bottom[1]->count() / bottom[1]->num();
  const int gt_dim = bottom[2]->count() / bottom[2]->num();
  const int nthreads = num * group_num_ * spatial_dim;
  
  DetectionGroupAccuracyParameter detect_acc_param 
          = this->layer_param_.detection_group_accuracy_param();
  const float field_whr = detect_acc_param.field_whr();
  const float field_xyr = detect_acc_param.field_xyr();
  const float bg_threshold = detect_acc_param.bg_threshold();
  
  // extract gt data
  for (int i = 0; i < num*group_num_; i++) {
    caffe_copy(spatial_dim, bottom_gt_data+i*gt_dim, 
            gt_label_.mutable_gpu_data()+i*spatial_dim);
    caffe_copy(bbox_dim, bottom_gt_data+i*gt_dim+spatial_dim, 
            gt_bbox_.mutable_gpu_data()+i*bbox_dim);
    caffe_copy(spatial_dim, bottom_gt_data+i*gt_dim+spatial_dim+bbox_dim, 
            gt_overlap_.mutable_gpu_data()+i*spatial_dim);
  }
  const Dtype* gt_label_data = gt_label_.gpu_data();
  const Dtype* gt_overlap_data = gt_overlap_.gpu_data();
 
  // The accuracy forward pass 
  CHECK_EQ(gt_bbox_.count(),4*nthreads);
  Dtype* accuracies = gt_bbox_.mutable_gpu_diff();
  Dtype* fore_accuracies = gt_bbox_.mutable_gpu_diff()+nthreads;
  Dtype* acc_counts = gt_bbox_.mutable_gpu_diff()+2*nthreads;
  Dtype* fore_counts = gt_bbox_.mutable_gpu_diff()+3*nthreads;
  caffe_gpu_set(nthreads,Dtype(0),accuracies);
  caffe_gpu_set(nthreads,Dtype(0),fore_accuracies);
  caffe_gpu_set(nthreads,Dtype(0),acc_counts);
  caffe_gpu_set(nthreads,Dtype(0),fore_counts);

  DetGroupAccForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, conf_data, gt_label_data, 
      gt_overlap_data, spatial_dim, conf_dim, cls_num_, bg_threshold,
      objectness_, has_ignore_label_, ignore_label_, acc_counts, fore_counts,
      accuracies, fore_accuracies);

  Dtype accuracy, fore_accuracy, acc_count, fore_count;
  caffe_gpu_asum(nthreads, accuracies, &accuracy);
  caffe_gpu_asum(nthreads, fore_accuracies, &fore_accuracy); 
  caffe_gpu_asum(nthreads, acc_counts, &acc_count);
  caffe_gpu_asum(nthreads, fore_counts, &fore_count);

  if (acc_count != 0) {
    accuracy /= acc_count;
  } else {
    accuracy = Dtype(-1);
  }
  if (fore_count != 0) {
    fore_accuracy /= fore_count;
  } else {
    fore_accuracy = Dtype(-1);
  }
  
  // The bbox iou forward pass 
  Dtype* bbox_ious = gt_bbox_.mutable_gpu_diff();
  Dtype* iou_counts = gt_bbox_.mutable_gpu_diff()+nthreads;
  caffe_gpu_set(nthreads,Dtype(0),bbox_ious);
  caffe_gpu_set(nthreads,Dtype(0),iou_counts);
  const Dtype* bbox_mean_data = bbox_mean_.gpu_data();
  const Dtype* bbox_std_data = bbox_std_.gpu_data();
  const Dtype* field_ws_data = field_ws_.gpu_data();
  const Dtype* field_hs_data = field_hs_.gpu_data();
  const Dtype* gt_bbox_data = gt_bbox_.gpu_data();

  DetGroupBboxIouForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, gt_label_data, gt_bbox_data,
      group_num_, width, height, bbox_mean_data, bbox_std_data, field_ws_data, 
      field_hs_data, stride_, field_whr, field_xyr, has_ignore_label_, ignore_label_, 
      iou_counts, bbox_ious);
  
  Dtype iou_count, bbox_iou;
  caffe_gpu_asum(nthreads, iou_counts, &iou_count);
  caffe_gpu_asum(nthreads, bbox_ious, &bbox_iou); 
  if (iou_count != 0) { 
    bbox_iou /= iou_count;
  } else {
    bbox_iou = Dtype(-1);
  }

  DLOG(INFO) << "Acc = "<<accuracy<<", ForeAcc = "<<fore_accuracy<<", IOU = "<<bbox_iou;
  top[0]->mutable_cpu_data()[0] = accuracy;
  top[0]->mutable_cpu_data()[1] = fore_accuracy;
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = bbox_iou;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionGroupAccuracyLayer);

}  // namespace caffe
