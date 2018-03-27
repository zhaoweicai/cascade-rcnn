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
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/box_group_output_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void BoxGroupOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BoxGroupOutputParameter box_group_output_param 
          = this->layer_param_.box_group_output_param();
  fg_thr_ = box_group_output_param.fg_thr();
  iou_thr_ = box_group_output_param.iou_thr();
  nms_type_ = box_group_output_param.nms_type();
  output_proposal_with_score_ = (top.size() == 2);
  
  CHECK_EQ(bottom.size()%2,0); CHECK_GE(bottom.size(),2);
  bottom_num_ = bottom.size()/2;
  CHECK_EQ(bottom_num_,box_group_output_param.box_group_param_size());
  for (int i = 0; i < bottom_num_; i++) {
    BoxGroupParameter box_group_param = box_group_output_param.box_group_param(i);
    group_strides_.push_back(box_group_param.stride());
    const int group_num = box_group_param.field_h_size();
    CHECK_EQ(group_num,box_group_param.field_w_size());
    group_nums_.push_back(group_num);
    shared_ptr<Blob<Dtype> > field_ws_pointer(new Blob<Dtype>());
    field_ws_pointer->Reshape(group_num, 1, 1, 1);
    shared_ptr<Blob<Dtype> > field_hs_pointer(new Blob<Dtype>());
    field_hs_pointer->Reshape(group_num, 1, 1, 1);
    for (int j = 0; j < group_num; j++) {
      field_ws_pointer->mutable_cpu_data()[j] = box_group_param.field_w(j);
      field_hs_pointer->mutable_cpu_data()[j] = box_group_param.field_h(j);
    }
    group_field_ws_.push_back(field_ws_pointer);
    group_field_hs_.push_back(field_hs_pointer);
    
    shared_ptr<Blob<Dtype> > preds_blob_pointer(new Blob<Dtype>());
    preds_blob_pointer->ReshapeLike(*(bottom[i*2+1]));
    bbox_preds_.push_back(preds_blob_pointer);
  }
  
  // bbox bounds [min_xyr, max_xyr, min_whr, max_whr]
  bbox_bound_.Reshape(4,1,1,1);
  Dtype* bbox_bound_data = bbox_bound_.mutable_cpu_data();
  float field_whr = box_group_output_param.field_whr();
  float field_xyr = box_group_output_param.field_xyr();
  bbox_bound_data[0] = Dtype(-1)/field_xyr;
  bbox_bound_data[1] = Dtype(1)/field_xyr;
  bbox_bound_data[2] = log(Dtype(1)/field_whr); 
  bbox_bound_data[3] = log(Dtype(field_whr));
  
  // bbox mean and std
  bbox_mean_.Reshape(4,1,1,1); bbox_std_.Reshape(4,1,1,1);
  Dtype* bbox_mean_data = bbox_mean_.mutable_cpu_data();
  Dtype* bbox_std_data = bbox_std_.mutable_cpu_data();
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    int num_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_means,4); CHECK_EQ(num_stds,4);
    for (int i = 0; i < 4; i++) {
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
void BoxGroupOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size()%2,0); CHECK_GE(bottom.size(),2);
  CHECK_EQ(bottom_num_,bottom.size()/2);
  const int cls_num = bottom[0]->channels();
  const int num = bottom[0]->num() / group_nums_[0];
  for (int i = 0; i < bottom_num_; i++) {
    CHECK_EQ(num, bottom[i*2]->num() / group_nums_[i]);
    CHECK_EQ(cls_num,bottom[i*2]->channels());
    CHECK_EQ(4,bottom[i*2+1]->channels());
    CHECK_EQ(bottom[i*2]->num(),bottom[i*2+1]->num());
    CHECK_EQ(bottom[i*2]->width(),bottom[i*2+1]->width());
    CHECK_EQ(bottom[i*2]->height(),bottom[i*2+1]->height());
    bbox_preds_[i]->ReshapeLike(*(bottom[i*2+1]));
  }
  //dummy output reshape
  top[0]->Reshape(1, 5, 1, 1);
  if (output_proposal_with_score_) {
    top[1]->Reshape(1, 6, 1, 1);
  }
}

template <typename Dtype>
void BoxGroupOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num() / group_nums_[0];
  BoxGroupOutputParameter box_group_output_param 
          = this->layer_param_.box_group_output_param();
  const int max_nms_num = box_group_output_param.max_nms_num(); 
  const int max_post_nms_num = box_group_output_param.max_post_nms_num(); 
  const float min_size = box_group_output_param.min_size();
  
  // decode all bbox predictions and scores
  for (int j = 0; j < bottom_num_; j++) {
    const Dtype* bbox_data = bottom[j*2+1]->cpu_data();
    const int width = bottom[j*2+1]->width(), height = bottom[j*2+1]->height();
    Dtype* bbox_pred_data = bbox_preds_[j]->mutable_cpu_data();
    const int bbox_dim = bottom[j*2+1]->count() / bottom[j*2+1]->num();
    // decode bbox precidtions
    const Dtype* field_ws_data = group_field_ws_[j]->cpu_data();
    const Dtype* field_hs_data = group_field_hs_[j]->cpu_data();
    DecodeGroupBBoxes(bbox_data,num,group_nums_[j],bbox_dim,width,height,
            bbox_mean_.cpu_data(),bbox_std_.cpu_data(),bbox_bound_.cpu_data(),
            field_ws_data,field_hs_data,group_strides_[j],bbox_dim,bbox_pred_data);
  }

  vector<vector<BBox> > batch_boxes(num);
  int num_batch_boxes = 0;
  for (int i = 0; i < num; i++) {
    vector<BBox> bboxes;
    for (int j = 0; j < bottom_num_; j++) {
      const Dtype* prob_data = bottom[j*2]->cpu_data();
      const Dtype* bbox_pred_data = bbox_preds_[j]->cpu_data();
      const int group_num =  group_nums_[j];
      const int prob_dim = bottom[j*2]->count() / bottom[j*2]->num();
      const int bbox_dim = bbox_preds_[j]->count() / bbox_preds_[j]->num();
      const int width = bottom[j*2]->width(), height = bottom[j*2]->height();
      const int spatial_dim = width*height;
      const int img_width = width*group_strides_[j];
      const int img_height = height*group_strides_[j];
           
      // collecting proposals
      for (int k = 0; k < group_num; k++) {
        const int group_idx = i*group_num+k;
        for (int id = 0; id < spatial_dim; id++) {
          Dtype fg_score = 1-prob_data[group_idx*prob_dim+id]; // objectness score
          if (fg_score >= fg_thr_) {
            const int idx = group_idx*bbox_dim+id;
            BBox bbox = InitBBox(bbox_pred_data[idx], bbox_pred_data[spatial_dim+idx], 
                    bbox_pred_data[2*spatial_dim+idx], bbox_pred_data[3*spatial_dim+idx]); 
            bbox.score = fg_score;
            ClipBBox(bbox, img_width, img_height);
            if ((bbox.xmax-bbox.xmin+1) >= min_size 
                    && (bbox.ymax-bbox.ymin+1) >= min_size) {
              bboxes.push_back(bbox);
            }
          }
        }
      }
    }

    if (bboxes.size() > 10000) {
      LOG(INFO) << "The number of boxes before NMS: " << bboxes.size();
    }
    if (bboxes.size()<=0) continue;
    
    //ranking decreasingly
    int keep_num = bboxes.size();
    if (max_nms_num > 0 && bboxes.size() > max_nms_num) {
      keep_num = max_nms_num;
    }
    std::partial_sort(bboxes.begin(), bboxes.begin() + keep_num,
          bboxes.end(), SortBBoxDescend);
    bboxes.resize(keep_num);

    //NMS
    vector<BBox> new_bboxes;
    new_bboxes = ApplyNMS(bboxes, iou_thr_, true, nms_type_);
    int num_new_boxes = new_bboxes.size();
    if (max_post_nms_num > 0 && num_new_boxes > max_post_nms_num) {
      num_new_boxes = max_post_nms_num;
    }
    new_bboxes.resize(num_new_boxes);
    batch_boxes[i] = new_bboxes;
    num_batch_boxes += num_new_boxes;
  }

  // output rois [batch_idx x1 y1 x2 y2] for roi_pooling layer
  if (num_batch_boxes <= 0) {
    // for special case when there is no box
    top[0]->Reshape(1, 5, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    top_boxes[0]=0; top_boxes[1]=1; top_boxes[2]=1; 
    top_boxes[3]=10; top_boxes[4]=10;
  } else {
    const int top_dim = 5;
    top[0]->Reshape(num_batch_boxes, top_dim, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    int idx = 0; 
    for (int i = 0; i < batch_boxes.size(); i++) {
      const vector<BBox> bboxes = batch_boxes[i];
      for (int j = 0; j < batch_boxes[i].size(); j++) {
        top_boxes[idx*top_dim] = i;
        top_boxes[idx*top_dim+1] = bboxes[j].xmin;
        top_boxes[idx*top_dim+2] = bboxes[j].ymin;
        top_boxes[idx*top_dim+3] = bboxes[j].xmax;
        top_boxes[idx*top_dim+4] = bboxes[j].ymax;
        idx++;
      }
    }
    CHECK_EQ(idx,num_batch_boxes);
  }
  // output proposals+scores [batch_idx x1 y1 x2 y2 score] for proposal detection
  if (output_proposal_with_score_) {
    if (num_batch_boxes <= 0) {
      // for special case when there is no box
      top[1]->Reshape(1, 6, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      caffe_set(top[1]->count(), Dtype(0), top_boxes_scores); 
    } else {
      const int top_dim = 6;
      top[1]->Reshape(num_batch_boxes, top_dim, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      int idx = 0;
      for (int i = 0; i < batch_boxes.size(); i++) {
        const vector<BBox> bboxes = batch_boxes[i];
        for (int j = 0; j < batch_boxes[i].size(); j++) {
          top_boxes_scores[idx*top_dim] = i;
          top_boxes_scores[idx*top_dim+1] = bboxes[j].xmin;
          top_boxes_scores[idx*top_dim+2] = bboxes[j].ymin;
          top_boxes_scores[idx*top_dim+3] = bboxes[j].xmax;
          top_boxes_scores[idx*top_dim+4] = bboxes[j].ymax;
          top_boxes_scores[idx*top_dim+5] = bboxes[j].score;
          idx++;
        }
      }
      CHECK_EQ(idx,num_batch_boxes);
    }
  }
}

INSTANTIATE_CLASS(BoxGroupOutputLayer);
REGISTER_LAYER_CLASS(BoxGroupOutput);

}  // namespace caffe
