// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <cfloat>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "caffe/layers/decode_bbox_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void DecodeBBoxLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // bbox mean and std
  BBoxRegParameter bbox_reg_param = this->layer_param_.bbox_reg_param();
  bbox_mean_.Reshape(4,1,1,1); bbox_std_.Reshape(4,1,1,1);
  if (bbox_reg_param.bbox_mean_size() > 0 && bbox_reg_param.bbox_std_size() > 0) {
    int num_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_means,4); CHECK_EQ(num_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_mean_.mutable_cpu_data()[i] = bbox_reg_param.bbox_mean(i);
      bbox_std_.mutable_cpu_data()[i] = bbox_reg_param.bbox_std(i);
      CHECK_GT(bbox_std_.mutable_cpu_data()[i],0);
    }
  } else {
    caffe_set(bbox_mean_.count(), Dtype(0), bbox_mean_.mutable_cpu_data());
    caffe_set(bbox_std_.count(), Dtype(1), bbox_std_.mutable_cpu_data());
  }
}

template <typename Dtype>
void DecodeBBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  // bottom: bbox_blob, prior_blob, (match_gt_boxes)
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  if (bottom.size()>=3) {
    CHECK_EQ(bottom[0]->num(),bottom[2]->num());
    CHECK(this->phase_ == TRAIN);
  }
  CHECK_EQ(bottom[0]->channels(),8); 
  CHECK_EQ(bottom[1]->channels(),5); 
  bbox_pred_.ReshapeLike(*bottom[0]);
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void DecodeBBoxLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int bbox_dim = bottom[0]->channels();
  const int prior_dim = bottom[1]->channels();
  
  //decode prior box [img_id x1 y1 x2 y2]
  const Dtype* prior_data = bottom[1]->cpu_data(); 
  vector<BBox> prior_bboxes;
  for (int i = 0; i < num; i++) {
    BBox bbox;
    bbox.xmin = prior_data[i*prior_dim + 1];
    bbox.ymin = prior_data[i*prior_dim + 2];
    bbox.xmax = prior_data[i*prior_dim + 3];
    bbox.ymax = prior_data[i*prior_dim + 4];
    prior_bboxes.push_back(bbox);
  }
   
  // decode bbox predictions
  const Dtype* bbox_data = bottom[0]->cpu_data();
  Dtype* bbox_pred_data = bbox_pred_.mutable_cpu_data();
  
  DecodeBBoxesWithPrior(bbox_data,prior_bboxes,bbox_dim,bbox_mean_.cpu_data(),
          bbox_std_.cpu_data(),bbox_pred_data);
  
  vector<bool> valid_bbox_flags(num,true);
  // screen out mal-boxes
  if (this->phase_ == TRAIN) {
    for (int i = 0; i < num; i++) {
      const int base_index = i*bbox_dim+4;
      if (bbox_pred_data[base_index] > bbox_pred_data[base_index+2] 
              || bbox_pred_data[base_index+1] > bbox_pred_data[base_index+3]) {
        valid_bbox_flags[i] = false;
      }
    }
  } 
  // screen out high IoU boxes, to remove redundant gt boxes
  if (bottom.size()==3 && this->phase_ == TRAIN) {
    const Dtype* match_gt_boxes = bottom[2]->cpu_data();
    const int gt_dim = bottom[2]->channels();
    const float gt_iou_thr = this->layer_param_.decode_bbox_param().gt_iou_thr();
    for (int i = 0; i < num; i++) {
      const float overlap = match_gt_boxes[i*gt_dim+gt_dim-1];
      if (overlap >= gt_iou_thr) {
        valid_bbox_flags[i] = false;
      }
    }
  }
  
  vector<int> valid_bbox_ids;
  for (int i = 0; i < num; i++) {
    if (valid_bbox_flags[i]) {
      valid_bbox_ids.push_back(i);
    }
  }
  const int keep_num = valid_bbox_ids.size();
  CHECK_GT(keep_num,0);
  
  top[0]->Reshape(keep_num, prior_dim, 1, 1);
  Dtype* decoded_bbox_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < keep_num; i++) {
    const int keep_id = valid_bbox_ids[i];
    const int base_index = keep_id*bbox_dim+4;
    decoded_bbox_data[i*prior_dim] =  prior_data[keep_id*prior_dim];
    decoded_bbox_data[i*prior_dim+1] = bbox_pred_data[base_index]; 
    decoded_bbox_data[i*prior_dim+2] = bbox_pred_data[base_index+1]; 
    decoded_bbox_data[i*prior_dim+3] = bbox_pred_data[base_index+2]; 
    decoded_bbox_data[i*prior_dim+4] = bbox_pred_data[base_index+3];
  }
}

INSTANTIATE_CLASS(DecodeBBoxLayer);
REGISTER_LAYER_CLASS(DecodeBBox);

}  // namespace caffe
