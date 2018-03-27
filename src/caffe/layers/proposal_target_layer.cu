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
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/proposal_target_layer.hpp"

namespace caffe {

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  //parameters
  ProposalTargetParameter proposal_target_param = 
          this->layer_param_.proposal_target_param();
  const float fg_fraction = proposal_target_param.fg_fraction();
  const int num_img_per_batch = proposal_target_param.num_img_per_batch();
  const float fg_thr = proposal_target_param.fg_thr();
  const float bg_thr_hg = proposal_target_param.bg_thr_hg();
  const float bg_thr_lw = proposal_target_param.bg_thr_lw();
  CHECK_GE(fg_thr,bg_thr_hg);
  const int img_width = proposal_target_param.img_width();
  const int img_height = proposal_target_param.img_height();
  const bool iou_weighted = proposal_target_param.iou_weighted();
  const bool pickup_discard = proposal_target_param.pickup_discard();
  const bool append_gtbox = proposal_target_param.append_gtbox();
  
  // bbox mean and std
  BBoxRegParameter bbox_reg_param = this->layer_param_.bbox_reg_param();
  bool do_bbox_norm = false;
  vector<float> bbox_means, bbox_stds;
  if (bbox_reg_param.bbox_mean_size() > 0 && bbox_reg_param.bbox_std_size() > 0) {
    do_bbox_norm = true;
    int num_bbox_means = bbox_reg_param.bbox_mean_size();
    int num_bbox_stds = bbox_reg_param.bbox_std_size();
    CHECK_EQ(num_bbox_means,4); CHECK_EQ(num_bbox_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_means.push_back(bbox_reg_param.bbox_mean(i));
      bbox_stds.push_back(bbox_reg_param.bbox_std(i));
    }
  }
  
  //inputs
  int num_rois = bottom[0]->num();
  int num_gt_boxes = bottom[1]->num();
  //[img_id, x1, y1, x2, y2]
  const Dtype* rois_boxes_data = bottom[0]->cpu_data();
  const int rois_dim = bottom[0]->channels();
  //[img_id, x1, y1, x2, y2, label, ignored]
  caffe_copy(bottom[1]->count(),bottom[1]->gpu_data(),gt_boxes_.mutable_gpu_data());
  const Dtype* gt_boxes_data = gt_boxes_.cpu_data(); 
  const int gt_dim = bottom[1]->channels();

  //[img_id x1 y1 w h]
  vector<vector<Dtype> > rois_boxes, gt_boxes;
  for (int i = 0; i < num_rois; i++) {
    vector<Dtype> bb(rois_dim);
    bb[0] = rois_boxes_data[i*rois_dim]; bb[1] = rois_boxes_data[i*rois_dim+1];
    bb[2] = rois_boxes_data[i*rois_dim+2]; 
    bb[3] = rois_boxes_data[i*rois_dim+3]-bb[1]+1;
    bb[4] = rois_boxes_data[i*rois_dim+4]-bb[2]+1;
    CHECK_LT(bb[0],num_img_per_batch);
    rois_boxes.push_back(bb);
  }

  vector<Dtype> gt_labels, gt_ignored;
  for (int i = 0; i < num_gt_boxes; i++) {
    vector<Dtype> bb(rois_dim);
    int item_id = gt_boxes_data[i*gt_dim];
    if (item_id == -1) {
      continue;
    }
    bb[0] = gt_boxes_data[i*gt_dim];
    bb[1] = gt_boxes_data[i*gt_dim+1];
    bb[2] = gt_boxes_data[i*gt_dim+2]; 
    bb[3] = gt_boxes_data[i*gt_dim+3]-bb[1]+1;
    bb[4] = gt_boxes_data[i*gt_dim+4]-bb[2]+1;
    CHECK_GE(item_id,0);
    CHECK_LT(item_id,num_img_per_batch);
    gt_boxes.push_back(bb);
    gt_labels.push_back(gt_boxes_data[i*gt_dim+5]);
    gt_ignored.push_back(gt_boxes_data[i*gt_dim+6]);
    //append gt boxes to the end of rois
    if (append_gtbox) {
      rois_boxes.push_back(bb);
    }
  }
  num_gt_boxes = gt_boxes.size();
  num_rois = rois_boxes.size();
  
  // find the matched gt for each roi bb
  vector<int> max_gt_inds(num_rois); vector<float> max_overlaps(num_rois);
  for (int i = 0; i < num_rois; i++) {
    float maxop = -FLT_MAX; int maxid = -1; bool exist_gt = false;
    for (int j = 0; j < num_gt_boxes; j++) {
      if (gt_boxes[j][0] != rois_boxes[i][0]) continue;
      exist_gt = true;
      float op = JaccardOverlap(rois_boxes[i][1], rois_boxes[i][2], rois_boxes[i][3], rois_boxes[i][4],
                 gt_boxes[j][1], gt_boxes[j][2], gt_boxes[j][3], gt_boxes[j][4], "IOU"); 
      if (op > maxop) {
        maxop = op; maxid = j;
      }
    }
    if (exist_gt) {
      max_gt_inds[i] = maxid; max_overlaps[i] = maxop;
    } else {
      max_gt_inds[i] = -1; max_overlaps[i] = 0;
    }
  }
  
  //select foreground rois with overlap >= fg_thr
  //select background rois with overlap within [fg_thr_lw,fg_thr_hg]
  vector<std::pair<int,int> > fg_inds_gtids, bg_inds_gtids; 
  vector<std::pair<int,int> > discard_bg_inds_gtids, keep_inds_gtids;
  for (int i = 0; i < num_rois; i++) {
    if (max_overlaps[i] >= fg_thr) {
      CHECK_GT(gt_labels[max_gt_inds[i]],0); //check if fg?
      if (gt_ignored[max_gt_inds[i]]) continue; //ignored
      fg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    } else if (max_overlaps[i]>=bg_thr_lw && max_overlaps[i]<bg_thr_hg) {
      bg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    } else {
      discard_bg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    }
  }
  
  // decide the sampling batch size
  int actual_batch_size, fg_rois_per_batch;
  if (batch_size_ == -1) {
    // sampling all positive and negative rois
    actual_batch_size = fg_inds_gtids.size()+bg_inds_gtids.size();
    fg_rois_per_batch = fg_inds_gtids.size();
  } else {
    actual_batch_size = batch_size_;
    fg_rois_per_batch = round(fg_fraction*batch_size_);
  }
  
  int fg_rois_this_batch = std::min(fg_rois_per_batch,int(fg_inds_gtids.size()));
  if (fg_inds_gtids.size() > fg_rois_this_batch) {
    //random sampling
    caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
    shuffle(fg_inds_gtids.begin(), fg_inds_gtids.end(), shuffle_rng);
    fg_inds_gtids.resize(fg_rois_this_batch);
  }
  int bg_rois_this_batch = actual_batch_size-fg_rois_this_batch;
  bg_rois_this_batch = std::min(bg_rois_this_batch,int(bg_inds_gtids.size()));
  if (bg_inds_gtids.size() > (actual_batch_size-fg_rois_this_batch)) {
    //random sampling
    caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
    shuffle(bg_inds_gtids.begin(), bg_inds_gtids.end(), shuffle_rng);
    bg_inds_gtids.resize(bg_rois_this_batch);
  } else if (pickup_discard && discard_bg_inds_gtids.size()>0) {
    //pick up some samples from discarded pool
    int num_refind_bg_rois = actual_batch_size-fg_rois_this_batch-bg_inds_gtids.size();
    num_refind_bg_rois = std::min(num_refind_bg_rois,int(discard_bg_inds_gtids.size()));
    for (int i = 0; i < num_refind_bg_rois; i++) {
      bg_inds_gtids.push_back(discard_bg_inds_gtids[i]);
      bg_rois_this_batch++;
    }
  }
  
  int num_keep_rois = fg_rois_this_batch+bg_rois_this_batch;
  if ((num_keep_rois<actual_batch_size/2.0) || (bg_rois_this_batch==0)) {
    DLOG(INFO) <<"layer: "<<this->layer_param().name()<< ", sampled fg rois: " << fg_rois_this_batch 
            << ", sampled bg rois: "<<bg_rois_this_batch;
  }
  if (num_keep_rois == 0) {
    int num_backup = 1;
    LOG(INFO) << "sampled rois: " << num_keep_rois << ", backup rois: "<<num_backup;
    //collect num_backup random bg boxes
    vector<vector<Dtype> > backup_boxes;
    while (backup_boxes.size() < num_backup) {
      int img_id = caffe_rng_rand() % num_img_per_batch;
      int bb_x = caffe_rng_rand() % (img_width-32);
      int bb_y = caffe_rng_rand() % (img_height-32);
      int bb_width = caffe_rng_rand() % (img_width-bb_x);
      int bb_height = caffe_rng_rand() % (img_height-bb_y);
      bb_width = std::max(bb_width,32); bb_height = std::max(bb_height,32);
      float maxop = -FLT_MAX;
      for (int j = 0; j < num_gt_boxes; j++) {
        if (gt_boxes[j][0] != img_id) continue;
        float op = JaccardOverlap(Dtype(bb_x), Dtype(bb_y), Dtype(bb_width), Dtype(bb_height), 
                gt_boxes[j][1], gt_boxes[j][2], gt_boxes[j][3], gt_boxes[j][4], "IOU"); 
        if (op > maxop) maxop = op;
      }
      if (maxop >= fg_thr) continue;
      vector<Dtype> bb(5);
      bb[0] = img_id; bb[1] = bb_x; bb[2] = bb_y; bb[3] = bb_width; bb[4] = bb_height;
      backup_boxes.push_back(bb);
    }
    for (int i = 0; i < num_backup; i++) {
      rois_boxes.push_back(backup_boxes[i]);
      int tmp_roi_id = rois_boxes.size()-1;
      bg_inds_gtids.push_back(std::make_pair(tmp_roi_id,-1));
      bg_rois_this_batch++;
    }
    num_keep_rois += num_backup;
  }
  
  //append index and labels
  vector<Dtype> labels; 
  for (int i = 0; i < fg_rois_this_batch; i++) {
    keep_inds_gtids.push_back(fg_inds_gtids[i]);
    int tmplabel = gt_labels[fg_inds_gtids[i].second];
    CHECK_GT(tmplabel,0); labels.push_back(tmplabel);
  }
  for (int i = 0; i < bg_rois_this_batch; i++) {
    keep_inds_gtids.push_back(bg_inds_gtids[i]);
    labels.push_back(0);
  }
  
  //get the box regression target
  vector<vector<Dtype> > bbox_targets, match_gt_boxes;
  for (int i = 0; i < num_keep_rois; i++) {
    Dtype bb_width, bb_height, bb_ctr_x, bb_ctr_y;
    Dtype gt_width, gt_height, gt_ctr_x, gt_ctr_y;
    Dtype targets_dx, targets_dy, targets_dw, targets_dh;
    vector<Dtype> bb_target(4); 
    vector<Dtype> match_gt_box(6);
    int bbid = keep_inds_gtids[i].first, gtid = keep_inds_gtids[i].second;  
    if (gtid >= 0) {
      bb_width = rois_boxes[bbid][3]; bb_height = rois_boxes[bbid][4];
      bb_ctr_x = rois_boxes[bbid][1]+0.5*bb_width;
      bb_ctr_y = rois_boxes[bbid][2]+0.5*bb_height;
      gt_width = gt_boxes[gtid][3]; gt_height = gt_boxes[gtid][4];
      gt_ctr_x = gt_boxes[gtid][1]+0.5*gt_width;
      gt_ctr_y = gt_boxes[gtid][2]+0.5*gt_height;
      targets_dx = (gt_ctr_x - bb_ctr_x) / bb_width;
      targets_dy = (gt_ctr_y - bb_ctr_y) / bb_height;
      targets_dw = log(gt_width / bb_width);
      targets_dh = log(gt_height / bb_height);
      bb_target[0]=targets_dx; bb_target[1]=targets_dy;
      bb_target[2]=targets_dw; bb_target[3]=targets_dh;
      
      // bbox normalization
      if (do_bbox_norm) {
        bb_target[0] -= bbox_means[0]; bb_target[1] -= bbox_means[1];
        bb_target[2] -= bbox_means[2]; bb_target[3] -= bbox_means[3];
        bb_target[0] /= bbox_stds[0]; bb_target[1] /= bbox_stds[1];
        bb_target[2] /= bbox_stds[2]; bb_target[3] /= bbox_stds[3];
      }
      
      //positives for bbox evaluation
      if (labels[i] > 0) { 
        match_gt_box[0] = labels[i];
        match_gt_box[1] = gt_boxes[gtid][1]; match_gt_box[3] = gt_boxes[gtid][1]+gt_width-1; 
        match_gt_box[2] = gt_boxes[gtid][2]; match_gt_box[4] = gt_boxes[gtid][2]+gt_height-1; 
        CHECK_LT(bbid,int(max_overlaps.size())); CHECK_GE(max_overlaps[bbid],fg_thr); 
        match_gt_box[5] = max_overlaps[bbid];    
      }
    }
    bbox_targets.push_back(bb_target);
    match_gt_boxes.push_back(match_gt_box);
  }

  //prepare the outputs
  // rois
  top[0]->Reshape(num_keep_rois, 5, 1, 1);
  Dtype* rois_data = top[0]->mutable_cpu_data();
  // labels
  top[1]->Reshape(num_keep_rois, 1, 1, 1);
  Dtype* labels_data = top[1]->mutable_cpu_data();
  // targets
  top[2]->Reshape(num_keep_rois, bbox_target_dim_, 1, 1);
  Dtype* targets_data = top[2]->mutable_cpu_data();
  caffe_set(top[2]->count(), Dtype(0), targets_data);  
  // box inside weights for box regression
  top[3]->Reshape(num_keep_rois, bbox_target_dim_, 1, 1);
  Dtype* box_inside_weights_data = top[3]->mutable_cpu_data();
  caffe_set(top[3]->count(), Dtype(0), box_inside_weights_data);  
  // box outside weights for box regression
  top[4]->Reshape(num_keep_rois, bbox_target_dim_, 1, 1);
  Dtype* box_outside_weights_data = top[4]->mutable_cpu_data();
  caffe_set(top[4]->count(), Dtype(0), box_outside_weights_data);  
  // matched gt boxes
  top[5]->Reshape(num_keep_rois, 6, 1, 1);
  Dtype* match_gt_boxes_data = top[5]->mutable_cpu_data();
  caffe_set(top[5]->count(), Dtype(0), match_gt_boxes_data); 
  // sample weights for softmax loss
  if (has_sample_weight_) {
    top[6]->Reshape(num_keep_rois, 1, 1, 1);
    Dtype* sample_weights_data = top[6]->mutable_cpu_data();
    caffe_set(top[6]->count(), Dtype(1), sample_weights_data);
    Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
    if (iou_weighted) {
      for (int i = 0; i < num_keep_rois; i++) {
        sample_weights_data[i] = labels[i]>0? match_gt_boxes[i][5]:1;
      }
    }
    for (int i = 0; i < num_keep_rois; i++) {
      if (labels[i]>0) pos_weight_sum += sample_weights_data[i]; 
      else neg_weight_sum += sample_weights_data[i]; 
    }
    for (int i = 0; i < num_keep_rois; i++) {
      if (labels[i] > 0) {
        if (pos_weight_sum != 0) 
          sample_weights_data[i] *= (fg_fraction*num_keep_rois/pos_weight_sum); 
      } else {
        if (pos_weight_sum != 0) 
          sample_weights_data[i] *= ((1-fg_fraction)*num_keep_rois/neg_weight_sum); 
      }
    }    
  }
  
  for (int i = 0; i < num_keep_rois; i++) {
    int cls_id = labels[i], rois_id = keep_inds_gtids[i].first;
    labels_data[i] = cls_id;
    //rois = (img_id, x1, y1, x2, y2)
    rois_data[i*rois_dim] = rois_boxes[rois_id][0];
    rois_data[i*rois_dim+1] = rois_boxes[rois_id][1];
    rois_data[i*rois_dim+2] = rois_boxes[rois_id][2];
    rois_data[i*rois_dim+3] = rois_boxes[rois_id][1]+rois_boxes[rois_id][3]-1;
    rois_data[i*rois_dim+4] = rois_boxes[rois_id][2]+rois_boxes[rois_id][4]-1;
    if (cls_id == 0) continue;
    if (!bbox_cls_aware_) {
      cls_id = std::min(1,cls_id);
    } 
    const float bbox_thr_hg = proposal_target_param.bbox_thr_hg();
    const float bbox_thr_lw = proposal_target_param.bbox_thr_lw();
    const float rois_overlap = match_gt_boxes[i][5];
    int start_id = i*bbox_target_dim_+cls_id*4;
    for (int j = 0; j < 4; j++) {
      targets_data[start_id+j] = bbox_targets[i][j];
      if (bbox_thr_lw <= rois_overlap && rois_overlap <= bbox_thr_hg) {
        box_inside_weights_data[start_id+j] = 1; //every dim has the same weight
        box_outside_weights_data[start_id+j] = 1; //every instance has the same weight
      } else {
        box_inside_weights_data[start_id+j] = 0;
        box_outside_weights_data[start_id+j] = 0;
      }
    }
    for (int j = 0; j < 6; j++) {
      // (label, x1, y1, x2, y2, overlap)
      match_gt_boxes_data[i*6+j] = match_gt_boxes[i][j];
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalTargetLayer);

}  // namespace caffe
