// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "caffe/layers/bbox_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype>
void BboxAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->num(),bottom[2]->num());
}

template <typename Dtype>
void BboxAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->num(),bottom[2]->num());
  bool bbox_cls_aware = this->layer_param_.bbox_reg_param().cls_aware();
  if (!bbox_cls_aware) {
    CHECK_EQ(8,bottom[0]->channels());
  }
  top[0]->Reshape(1, 1, 1, 1);
  if (top.size() >= 2) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void BboxAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype after_avg_iou = 0, pre_avg_iou = 0, pre_roi_avg_iou = 0;
  // 4*K
  const Dtype* bbox_preds = bottom[0]->cpu_data();
  const int pred_dim = bottom[0]->channels();
  // (batch_id, x1, y1, x2, y2)
  const Dtype* rois = bottom[1]->cpu_data();
  const int roi_dim = bottom[1]->channels();
  // (label, x1, y1, x2, y2, overlap)
  const Dtype* gt_boxes = bottom[2]->cpu_data();
  const int gt_dim = bottom[2]->channels();
  const int num = bottom[0]->num();
  
  // bbox mean and std
  bool bbox_cls_aware = this->layer_param_.bbox_reg_param().cls_aware();
  bool do_bbox_norm = false;
  vector<float> bbox_means, bbox_stds;
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    do_bbox_norm = true;
    int num_bbox_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_bbox_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_bbox_means,4); CHECK_EQ(num_bbox_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_means.push_back(this->layer_param_.bbox_reg_param().bbox_mean(i));
      bbox_stds.push_back(this->layer_param_.bbox_reg_param().bbox_std(i));
    }
  }
  
  int fg_count = 0;
  vector<bool> gt_flags(num,false);
  for (int i = 0; i < num; i++) {
    if (gt_boxes[i*gt_dim] > 0) { //label
      // ignore gt bounding box for iou evaluation
      if (gt_boxes[i*gt_dim+gt_dim-1]>0.975) {
        gt_flags[i] = true; continue;
      }
      fg_count++; pre_avg_iou += gt_boxes[i*gt_dim+gt_dim-1];
    }
  }
  
  // compute iou for each output box
  for (int i = 0; i < num; i++) {
    int label = static_cast<int>(gt_boxes[i*gt_dim]);
    if (label <= 0) continue;
    if (!bbox_cls_aware) {
      label = std::min(1,label);
    }
    Dtype pred_x, pred_y, pred_w, pred_h;
    pred_x = bbox_preds[i*pred_dim+label*4]; pred_y = bbox_preds[i*pred_dim+label*4+1];
    pred_w = bbox_preds[i*pred_dim+label*4+2]; pred_h = bbox_preds[i*pred_dim+label*4+3];
    
    // bbox de-normalization
    if (do_bbox_norm) {
      pred_x *= bbox_stds[0]; pred_y *= bbox_stds[1];
      pred_w *= bbox_stds[2]; pred_h *= bbox_stds[3];
      pred_x += bbox_means[0]; pred_y += bbox_means[1];
      pred_w += bbox_means[2]; pred_h += bbox_means[3];
    }
    
    Dtype roi_x, roi_y, roi_w, roi_h;
    roi_x = rois[i*roi_dim+1]; roi_w = rois[i*roi_dim+3]-roi_x+1; 
    roi_y = rois[i*roi_dim+2]; roi_h = rois[i*roi_dim+4]-roi_y+1;
    Dtype gt_x, gt_y, gt_w, gt_h;
    gt_x = gt_boxes[i*gt_dim+1]; gt_w = gt_boxes[i*gt_dim+3]-gt_x+1; 
    gt_y = gt_boxes[i*gt_dim+2]; gt_h = gt_boxes[i*gt_dim+4]-gt_y+1;
    Dtype ctr_x, ctr_y, tx, ty, tw, th;
    ctr_x = roi_x+0.5*roi_w; ctr_y = roi_y+0.5*roi_h;
    tx = pred_x*roi_w+ctr_x; ty = pred_y*roi_h+ctr_y;
    tw = roi_w*exp(pred_w); th = roi_h*exp(pred_h);
    tx = tx-tw/Dtype(2); ty = ty-th/Dtype(2);
    if (gt_flags[i]) continue;
    Dtype iou = JaccardOverlap(tx,ty,tw,th,gt_x,gt_y,gt_w,gt_h,"IOU");
    after_avg_iou += iou;
    Dtype roi_iou = JaccardOverlap(roi_x,roi_y,roi_w,roi_h,gt_x,gt_y,gt_w,gt_h,"IOU");
    pre_roi_avg_iou += roi_iou;
  }

  if (fg_count > 0) pre_avg_iou /= Dtype(fg_count);
  else pre_avg_iou = Dtype(-1);
  if (fg_count > 0) after_avg_iou /= Dtype(fg_count);
  else after_avg_iou = Dtype(-1);
  if (fg_count > 0) pre_roi_avg_iou /= Dtype(fg_count);
  else pre_roi_avg_iou = Dtype(-1);

  top[0]->mutable_cpu_data()[0] = after_avg_iou;
  if (top.size() > 1) {
    top[1]->mutable_cpu_data()[0] = pre_roi_avg_iou;
  }
}

INSTANTIATE_CLASS(BboxAccuracyLayer);
REGISTER_LAYER_CLASS(BboxAccuracy);

}  // namespace caffe
