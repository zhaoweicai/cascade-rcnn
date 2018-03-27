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
void BoxGroupOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num() / group_nums_[0];
  BoxGroupOutputParameter box_group_output_param 
          = this->layer_param_.box_group_output_param();
  const int max_nms_num = box_group_output_param.max_nms_num(); 
  const int max_post_nms_num = box_group_output_param.max_post_nms_num(); 
  const float min_size = box_group_output_param.min_size();
  
  // decode all bbox predictions and scores
  for (int j = 0; j < bottom_num_; j++) {
    const Dtype* bbox_data = bottom[j*2+1]->gpu_data();
    const int width = bottom[j*2+1]->width(), height = bottom[j*2+1]->height();
    Dtype* bbox_pred_data = bbox_preds_[j]->mutable_gpu_data();
    const int bbox_dim = bottom[j*2+1]->count() / bottom[j*2+1]->num();
    const int nthreads = num*group_nums_[j]*width*height;
    // decode bbox precidtions
    const Dtype* field_ws_data = group_field_ws_[j]->gpu_data();
    const Dtype* field_hs_data = group_field_hs_[j]->gpu_data();
    DecodeGroupBBoxesGPU(nthreads,bbox_data,group_nums_[j],bbox_dim,width,height,
            bbox_mean_.gpu_data(),bbox_std_.gpu_data(),bbox_bound_.gpu_data(),
            field_ws_data,field_hs_data,group_strides_[j],bbox_dim,bbox_pred_data);
  }

  vector<vector<BBox> > batch_boxes(num);
  int num_batch_boxes = 0;
  for (int i = 0; i < num; i++) {
    vector<BBox> bboxes;
    for (int j = 0; j < bottom_num_; j++) {
      const Dtype* prob_cpu = bottom[j*2]->cpu_data();
      const Dtype* bbox_pred_cpu = bbox_preds_[j]->cpu_data();
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
          Dtype fg_score = 1-prob_cpu[group_idx*prob_dim+id]; // objectness score
          if (fg_score >= fg_thr_) {
            const int idx = group_idx*bbox_dim+id;
            BBox bbox = InitBBox(bbox_pred_cpu[idx], bbox_pred_cpu[spatial_dim+idx], 
                    bbox_pred_cpu[2*spatial_dim+idx], bbox_pred_cpu[3*spatial_dim+idx]); 
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
      DLOG(INFO) << "The number of boxes before NMS: " << bboxes.size();
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

INSTANTIATE_LAYER_GPU_FUNCS(BoxGroupOutputLayer);

}  // namespace caffe
