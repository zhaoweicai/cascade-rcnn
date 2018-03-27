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

#include "caffe/util/bbox_util.hpp"
#include "caffe/layers/detection_group_accuracy_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void DetectionGroupAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DetectionGroupAccuracyParameter detect_acc_param 
          = this->layer_param_.detection_group_accuracy_param();
  group_num_ = detect_acc_param.field_h_size();
  CHECK_GE(group_num_,1);
  CHECK_EQ(group_num_,detect_acc_param.field_w_size());
  cls_num_ = detect_acc_param.cls_num();
  coord_num_ = 4;
  stride_ = detect_acc_param.stride();
  field_ws_.Reshape(group_num_,1,1,1); 
  field_hs_.Reshape(group_num_,1,1,1);
  for (int j = 0; j < group_num_; j++) {
    field_ws_.mutable_cpu_data()[j] = detect_acc_param.field_w(j);
    field_hs_.mutable_cpu_data()[j] = detect_acc_param.field_h(j);
  }
  
  objectness_ = detect_acc_param.objectness();
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  
  // bbox mean and std
  bbox_mean_.Reshape(4,1,1,1); bbox_std_.Reshape(4,1,1,1);
  Dtype* bbox_mean_data = bbox_mean_.mutable_cpu_data();
  Dtype* bbox_std_data = bbox_std_.mutable_cpu_data();
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    int num_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_means,coord_num_); CHECK_EQ(num_stds,coord_num_);
    for (int i = 0; i < coord_num_; i++) {
      bbox_mean_data[i] = this->layer_param_.bbox_reg_param().bbox_mean(i);
      bbox_std_data[i] = this->layer_param_.bbox_reg_param().bbox_std(i);
      CHECK_GT(bbox_std_data[i],0);
    }
  } else {
    caffe_set(bbox_mean_.count(), Dtype(0), bbox_mean_data);
    caffe_set(bbox_std_.count(), Dtype(1), bbox_std_data);
  }
}

template <typename Dtype>
void DetectionGroupAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num(); CHECK_EQ(0,num % group_num_);
  int height = bottom[0]->height(); int width = bottom[0]->width();
  CHECK_EQ(num,bottom[1]->num()); CHECK_EQ(num,bottom[2]->num());
  CHECK_EQ(height,bottom[1]->height()); CHECK_EQ(height,bottom[2]->height());
  CHECK_EQ(width,bottom[1]->width()); CHECK_EQ(width,bottom[2]->width());
  CHECK_EQ(cls_num_,bottom[0]->channels());
  CHECK_EQ(coord_num_,bottom[1]->channels());
  CHECK_EQ(coord_num_+2,bottom[2]->channels());
  if (objectness_) {
    CHECK_EQ(2,cls_num_);
  }
  top[0]->Reshape(1, 1, 1, 2);
  if (top.size() >= 2) {
    top[1]->Reshape(1, 1, 1, 1);
  }
  gt_bbox_.Reshape(num, coord_num_, height, width);
  gt_label_.Reshape(num, 1, height, width);
  gt_overlap_.Reshape(num, 1, height, width);
}

template <typename Dtype>
void DetectionGroupAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* conf_data = bottom[0]->cpu_data();
  const Dtype* bbox_data = bottom[1]->cpu_data();
  const Dtype* bottom_gt_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num() / group_num_;
  const int height = bottom[0]->height(); 
  const int width = bottom[0]->width();
  const int spatial_dim = height*width;
  const int conf_dim = bottom[0]->count() / bottom[0]->num();
  const int bbox_dim = bottom[1]->count() / bottom[1]->num();
  const int gt_dim = bottom[2]->count() / bottom[2]->num();

  DetectionGroupAccuracyParameter detect_acc_param 
          = this->layer_param_.detection_group_accuracy_param();
  const float field_whr = detect_acc_param.field_whr();
  const float field_xyr = detect_acc_param.field_xyr();
  const float bg_threshold = detect_acc_param.bg_threshold();
  
  // extract gt data
  for (int i = 0; i < num*group_num_; i++) {
    caffe_copy(spatial_dim, bottom_gt_data+i*gt_dim, 
            gt_label_.mutable_cpu_data()+i*spatial_dim);
    caffe_copy(bbox_dim, bottom_gt_data+i*gt_dim+spatial_dim, 
            gt_bbox_.mutable_cpu_data()+i*bbox_dim);
    caffe_copy(spatial_dim, bottom_gt_data+i*gt_dim+spatial_dim+bbox_dim, 
            gt_overlap_.mutable_cpu_data()+i*spatial_dim);
  }
  const Dtype* gt_label_data = gt_label_.cpu_data();
  const Dtype* gt_overlap_data = gt_overlap_.cpu_data();
 
  // The accuracy forward pass 
  Dtype accuracy = 0, fore_accuracy = 0;
  int acc_count = 0, fore_count = 0;
  for (int i = 0; i < num*group_num_; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int label_value = static_cast<int>(gt_label_data[i*spatial_dim+j]);
      if (objectness_) {
        label_value = std::min(1,label_value);
      }
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (label_value == 0 && gt_overlap_data[i*spatial_dim+j] >= bg_threshold) {
        continue; // ignored bounding boxes
      }
      if (label_value != 0) {
        ++fore_count;
      }
      ++acc_count;
      Dtype max_score = -FLT_MAX; int max_id = -1;
      for (int k = 0; k < cls_num_; k++) {
        if (conf_data[i*conf_dim+k*spatial_dim+j] > max_score) {
          max_score = conf_data[i*conf_dim+k*spatial_dim+j];
          max_id = k;
        }
      }
      if (max_id == label_value) {
        ++accuracy;
        if (label_value != 0) {
          ++fore_accuracy; 
        }
      }
    }
  }
  
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

  int iou_count = 0;
  Dtype bbox_iou = 0;
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;
  const Dtype* bbox_mean_data = bbox_mean_.cpu_data();
  const Dtype* bbox_std_data = bbox_std_.cpu_data();
  const Dtype* field_ws_data = field_ws_.cpu_data();
  const Dtype* field_hs_data = field_hs_.cpu_data();
  const Dtype* gt_bbox_data = gt_bbox_.cpu_data();

  for (int i = 0; i < num*group_num_; ++i) {
    const int k = i % group_num_;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int label_value = static_cast<int>(gt_label_data[i*spatial_dim+h*width+w]);
        if ((has_ignore_label_ && label_value == ignore_label_) || (label_value==0))  {
          continue;
        }
        Dtype tx, ty, tw, th; 
        const int bbox_idx = i*bbox_dim + h*width + w;
        tx = bbox_data[bbox_idx];
        ty = bbox_data[bbox_idx+spatial_dim];
        tw = bbox_data[bbox_idx+2*spatial_dim];
        th = bbox_data[bbox_idx+3*spatial_dim];
        
        // bbox de-normalization
        tx *= bbox_std_data[0]; ty *= bbox_std_data[1];
        tw *= bbox_std_data[2]; th *= bbox_std_data[3];
        tx += bbox_mean_data[0]; ty += bbox_mean_data[1];
        tw += bbox_mean_data[2]; th += bbox_mean_data[3];
        
        // bbox bounding
        tx = std::max(min_xyr,tx); tx = std::min(max_xyr,tx); 
        ty = std::max(min_xyr,ty); ty = std::min(max_xyr,ty);
        tw = std::max(min_whr,tw); tw = std::min(max_whr,tw); 
        th = std::max(min_whr,th); th = std::min(max_whr,th);
        
        tx = tx*field_ws_data[k] + (w+Dtype(0.5))*stride_;
        ty = ty*field_hs_data[k] + (h+Dtype(0.5))*stride_;
        tw = field_ws_data[k] * exp(tw); th = field_hs_data[k] * exp(th);
        tx = tx - tw/Dtype(2); ty = ty - th/Dtype(2);

        Dtype gx, gy, gw, gh;
        gx = gt_bbox_data[bbox_idx]; 
        gy = gt_bbox_data[bbox_idx+spatial_dim];
        gw = gt_bbox_data[bbox_idx+2*spatial_dim]; 
        gh = gt_bbox_data[bbox_idx+3*spatial_dim];
        gx = gx - gw/Dtype(2); gy = gy - gh/Dtype(2);
        
        Dtype iou = JaccardOverlap(tx,ty,tw,th,gx,gy,gw,gh,"IOU");
        bbox_iou += iou;
        iou_count++; 
      }
    }
  }
  
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

INSTANTIATE_CLASS(DetectionGroupAccuracyLayer);
REGISTER_LAYER_CLASS(DetectionGroupAccuracy);

}  // namespace caffe
