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
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layers/detection_group_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
  }
    
  DetectionGroupLossParameter detect_group_loss_param = 
          this->layer_param_.detection_group_loss_param();
  group_num_ = detect_group_loss_param.field_h_size();
  CHECK_GE(group_num_,1);
  CHECK_EQ(group_num_,detect_group_loss_param.field_w_size());
  cls_num_ = detect_group_loss_param.cls_num();
  coord_num_ = 4; 
  stride_ = detect_group_loss_param.stride();
  field_ws_.Reshape(group_num_,1,1,1); 
  field_hs_.Reshape(group_num_,1,1,1);
  for (int j = 0; j < group_num_; j++) {
    field_ws_.mutable_cpu_data()[j] = detect_group_loss_param.field_w(j);
    field_hs_.mutable_cpu_data()[j] = detect_group_loss_param.field_h(j);
  }
  lambda_ = detect_group_loss_param.lambda();
  bbox_loss_type_ = detect_group_loss_param.bbox_loss_type();
  objectness_ = detect_group_loss_param.objectness();
  pos_neg_weighted_ = detect_group_loss_param.pos_neg_weighted();
  bg_threshold_ = detect_group_loss_param.bg_threshold();
  bg_multiple_ = detect_group_loss_param.bg_multiple();
  neg_mining_type_ = detect_group_loss_param.neg_mining_type();
  batch_size_ = detect_group_loss_param.batch_size();
  min_num_neg_ = detect_group_loss_param.min_num_neg();
  neg_ranking_type_ = detect_group_loss_param.neg_ranking_type();

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  
  // reset the random generator seed
  const unsigned int shuffle_rng_seed = caffe_rng_rand();
  shuffle_rng_.reset(new Caffe::RNG(shuffle_rng_seed));
  
  int num = bottom[0]->num();
  int height = bottom[0]->height();  
  int width = bottom[0]->width();
    
  keep_map_.Reshape(num, 1, height, width);
  weight_map_.Reshape(num, 1, height, width);
  bbox_diff_.Reshape(num, coord_num_, height, width);
  gt_bbox_.Reshape(num, coord_num_, height, width);
  gt_label_.Reshape(num, 1, height, width);
  gt_overlap_.Reshape(num, 1, height, width);

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
  
  do_bound_bbox_ = false;
  if (detect_group_loss_param.has_field_xyr() && detect_group_loss_param.has_field_whr()) {
    do_bound_bbox_ = true;
    // bbox bounds [min_xyr, max_xyr, min_whr, max_whr]
    bbox_bound_.Reshape(4,1,1,1);
    Dtype* bbox_bound_data = bbox_bound_.mutable_cpu_data();
    Dtype field_xyr = detect_group_loss_param.field_xyr();
    Dtype field_whr = detect_group_loss_param.field_whr(); 
    Dtype min_xyr = -1.0/field_xyr, max_xyr = 1.0/field_xyr;
    Dtype min_whr = log(1.0/field_whr), max_whr = log(field_whr);
    Dtype xyr_mean = (bbox_mean_data[0]+bbox_mean_data[1])/2.0;
    Dtype whr_mean = (bbox_mean_data[2]+bbox_mean_data[3])/2.0;
    Dtype xyr_std = sqrt(bbox_std_data[0]*bbox_std_data[1]);
    Dtype whr_std = sqrt(bbox_std_data[2]*bbox_std_data[3]);
    min_xyr -= xyr_mean; max_xyr -= xyr_mean;
    min_whr -= whr_mean; max_whr -= whr_mean;
    min_xyr /= xyr_std; max_xyr /= xyr_std;
    min_whr /= whr_std; max_whr /= whr_std; 
    bbox_bound_data[0] = min_xyr; bbox_bound_data[1] = max_xyr;
    bbox_bound_data[2] = min_whr; bbox_bound_data[3] = max_whr;
  }
  
  // set up softmax layer
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::Reshape(
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
  
  LossLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(1,1,1,2);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  
  keep_map_.Reshape(num, 1, height, width);
  weight_map_.Reshape(num, 1, height, width);
  bbox_diff_.Reshape(num, coord_num_, height, width);
  gt_bbox_.Reshape(num, coord_num_, height, width);
  gt_label_.Reshape(num, 1, height, width);
  gt_overlap_.Reshape(num, 1, height, width);
}

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_gt_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num() / group_num_;
  const int height = bottom[0]->height(); 
  const int width = bottom[0]->width();
  const int spatial_dim = height*width;
  const int conf_dim = bottom[0]->count() / bottom[0]->num();
  const int bbox_dim = bottom[1]->count() / bottom[1]->num();
  const int gt_dim = bottom[2]->count() / bottom[2]->num();
  
  // extract gt data
  for (int i = 0; i < num; i++) {
    for (int nn = 0; nn < group_num_; nn++) {
      const int group_idx = i*group_num_+nn;
      caffe_copy(spatial_dim, bottom_gt_data+group_idx*gt_dim, 
              gt_label_.mutable_cpu_data()+group_idx*spatial_dim);
      caffe_copy(bbox_dim, bottom_gt_data+group_idx*gt_dim+spatial_dim, 
              gt_bbox_.mutable_cpu_data()+group_idx*bbox_dim);
      caffe_copy(spatial_dim, bottom_gt_data+group_idx*gt_dim+spatial_dim+bbox_dim, 
              gt_overlap_.mutable_cpu_data()+group_idx*spatial_dim);
    }
  }
  const Dtype* gt_label_data = gt_label_.cpu_data();
  const Dtype* gt_overlap_data = gt_overlap_.cpu_data();
  
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  
  // assign negatives ranking scores
  Dtype* neg_ranking_data = keep_map_.mutable_cpu_diff();
  caffe_set(keep_map_.count(), Dtype(0), neg_ranking_data);
  if ((neg_mining_type_ == "bootstrap") || (neg_mining_type_ == "mixture")) {
    for (int nn = 0; nn < group_num_; nn++) {
      if (neg_ranking_type_ == "conf_prob") {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < spatial_dim; j++) {
            neg_ranking_data[(i*group_num_+nn)*spatial_dim+j] 
                    = 1-prob_data[(i*group_num_+nn)*conf_dim+j];
          }
        } 
      } else if (neg_ranking_type_ == "conf_loss") {
        for (int i = 0; i < num; ++i) {
          const int group_idx = i*group_num_+nn;
          for (int j = 0; j < spatial_dim; j++) {
            int label_value = static_cast<int>(gt_label_data[group_idx*spatial_dim+j]);
            if (objectness_) {
              label_value = std::min(1,label_value);
            }
            CHECK_GE(label_value, 0);
            CHECK_LT(label_value, prob_.channels());
            neg_ranking_data[group_idx*spatial_dim+j] = -log(std::max(
                    prob_data[group_idx*conf_dim+label_value*spatial_dim+j],Dtype(FLT_MIN)));
          }
        }
      } else {
        LOG(FATAL) << "Unknown negative ranking type.";
      }
    }
  }
        
  //build the keep map
  Dtype* keep_map_data = keep_map_.mutable_cpu_data();
  caffe_set(keep_map_.count(), Dtype(0), keep_map_data);
  int total_neg_num = 0, total_pos_num = 0, total_sample_num = 0;
  
  for (int i = 0; i < num; ++i) {
    int num_pos, num_neg, sample_pos_num, sample_neg_num;
    // collecting potential positives and negatives
    std::vector<int> pos_keep_idxs, neg_keep_idxs;
    std::vector<std::pair<Dtype, int> > neg_score_idx_pairs;
    for (int nn = 0; nn < group_num_; nn++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = (i*group_num_+nn)*spatial_dim+j;
        const int label_value = static_cast<int>(gt_label_data[base_index]);
        const Dtype overlap = gt_overlap_data[base_index];
        if (label_value > 0) {
          pos_keep_idxs.push_back(nn*spatial_dim+j);
        } else if ((label_value==0) && (overlap<bg_threshold_)) {
          neg_score_idx_pairs.push_back(std::make_pair(neg_ranking_data[base_index], 
                  nn*spatial_dim+j));
        }
      }
    }
    num_pos = pos_keep_idxs.size();
    num_neg = neg_score_idx_pairs.size();
    total_pos_num += num_pos; total_neg_num += num_neg; 
    
    // decide sample numbers
    if (batch_size_ == -1) {
      sample_pos_num = num_pos;
      sample_neg_num = num_pos*bg_multiple_;
      sample_neg_num = std::max(sample_neg_num, min_num_neg_);
      sample_neg_num = std::min(sample_neg_num,num_neg);
    } else if (batch_size_ > 0) {
      sample_pos_num = round(float(batch_size_)/(1.0+bg_multiple_));
      sample_pos_num = std::min(sample_pos_num,num_pos);
      sample_neg_num = batch_size_-sample_pos_num;
      sample_neg_num = std::min(sample_neg_num,num_neg);
    } else {
      LOG(FATAL) << "Incorrect batch size!";
    }
    total_sample_num += (sample_pos_num+sample_neg_num);
    
    // random sample positives if there are too many
    if (num_pos > sample_pos_num) {
      caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
      shuffle(pos_keep_idxs.begin(), pos_keep_idxs.end(), shuffle_rng);
      pos_keep_idxs.resize(sample_pos_num);
    }

    // sample negatives if there are too many
    if (num_neg > sample_neg_num) {
      if (neg_mining_type_ == "random") {
        caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
        shuffle(neg_score_idx_pairs.begin(), neg_score_idx_pairs.end(), shuffle_rng);
        for (int j = 0; j < sample_neg_num; j++) {
          neg_keep_idxs.push_back(neg_score_idx_pairs[j].second);
        }
      } else if (neg_mining_type_ == "bootstrap") {
        std::partial_sort(neg_score_idx_pairs.begin(), neg_score_idx_pairs.begin()+sample_neg_num,
               neg_score_idx_pairs.end(), std::greater<std::pair<Dtype, int> >());
        for (int j = 0; j < sample_neg_num; j++) {
          neg_keep_idxs.push_back(neg_score_idx_pairs[j].second);
        }
      } else if (neg_mining_type_ == "mixture") {
        const int hard_num = round(sample_neg_num*0.5);
        std::partial_sort(neg_score_idx_pairs.begin(), neg_score_idx_pairs.begin()+hard_num,
               neg_score_idx_pairs.end(), std::greater<std::pair<Dtype, int> >());
        for (int j = 0; j < hard_num; j++) {
          neg_keep_idxs.push_back(neg_score_idx_pairs[j].second);
        }
        neg_score_idx_pairs.erase(neg_score_idx_pairs.begin(),neg_score_idx_pairs.begin()+hard_num);
        const int random_num = sample_neg_num-hard_num;
        if (random_num > 0) {
          caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
          shuffle(neg_score_idx_pairs.begin(), neg_score_idx_pairs.end(), shuffle_rng);
        }
        for (int j = 0; j < random_num; j++) {
          neg_keep_idxs.push_back(neg_score_idx_pairs[j].second);
        }
      } else {
        LOG(FATAL) << "Unknown negative sampling strategy!";
      }
    } else {
      for (int j = 0; j < num_neg; j++) {
        neg_keep_idxs.push_back(neg_score_idx_pairs[j].second);
      }
    }
    
    // assign keep samples to keep_map
    CHECK_EQ(sample_pos_num,pos_keep_idxs.size());
    CHECK_EQ(sample_neg_num,neg_keep_idxs.size());
    for (int j = 0; j < sample_pos_num; j++) {
      const int keep_idx = pos_keep_idxs[j];
      keep_map_data[i*group_num_*spatial_dim+keep_idx] = Dtype(1);
    }
    for (int j = 0; j < sample_neg_num; j++) {
      const int keep_idx = neg_keep_idxs[j];
      keep_map_data[i*group_num_*spatial_dim+keep_idx] = Dtype(1);
    } 
  }
  int total_sample_num_check = caffe_cpu_asum(keep_map_.count(),keep_map_data);
  CHECK_EQ(total_sample_num, total_sample_num_check);
  DLOG(INFO)<<"total positive = "<<total_pos_num <<", total negative = "
          <<total_neg_num<<", keep = "<<total_sample_num;
    
  //setup weight map
  Dtype* weight_map_data = weight_map_.mutable_cpu_data();
  caffe_set(weight_map_.count(), Dtype(1), weight_map_data);
  Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
  if (pos_neg_weighted_) {
    for (int i = 0; i < weight_map_.count(); i++) {
      const int label_value = static_cast<int>(gt_label_data[i]);
      if (keep_map_data[i] == 0) continue;
      if (label_value > 0) {
        pos_weight_sum += weight_map_data[i]; 
      } else {
        neg_weight_sum += weight_map_data[i]; 
      }
    }
    Dtype fg_weight = Dtype(1)/(1+bg_multiple_);
    for (int i = 0; i < weight_map_.count(); i++) {
      const int label_value = static_cast<int>(gt_label_data[i]);
      if (label_value > 0) {
        if (pos_weight_sum != 0) {
          weight_map_data[i] *= (fg_weight*total_sample_num/pos_weight_sum); 
        }
      } else {
        if (neg_weight_sum != 0) {
          weight_map_data[i] *= ((1-fg_weight)*total_sample_num/neg_weight_sum); 
        }
      }
    }
  }
    
  // computes the softmax loss.
  int conf_count = 0;
  Dtype conf_loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int nn = 0; nn < group_num_; nn++) {
      const int group_idx = i*group_num_+nn;
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = group_idx*spatial_dim+j;
        int label_value = static_cast<int>(gt_label_data[base_index]);
        if (objectness_) {
          label_value = std::min(1,label_value);
        }
        const int keep_flag = static_cast<int>(keep_map_data[base_index]); 
        if ((has_ignore_label_ && label_value == ignore_label_) || (keep_flag == 0)) {
          continue;
        } else {
          CHECK_GE(label_value, 0);
          CHECK_LT(label_value, prob_.channels());
          conf_loss -= log(std::max(prob_data[group_idx*conf_dim+label_value*spatial_dim+j],
                  Dtype(FLT_MIN))) * weight_map_data[base_index];
          conf_count++;
        }
      }
    }
  }
  
  // the forward pass computes euclidean loss 
  int bbox_count = 0;
  Dtype bbox_loss = 0;
  Dtype* bbox_diff_data = bbox_diff_.mutable_cpu_data();
  caffe_set(bbox_diff_.count(), Dtype(0), bbox_diff_data);
  
  // encode gt bbox
  EncodeGroupBBoxes(gt_bbox_.cpu_data(), gt_label_data, num, group_num_, width, 
          height, bbox_mean_.cpu_data(), bbox_std_.cpu_data(), field_ws_.cpu_data(), 
          field_hs_.cpu_data(), stride_, gt_bbox_.mutable_cpu_diff());
  
  // bound predictions
  if (do_bound_bbox_) {
    BoundBBoxPreds(bottom[1]->mutable_cpu_data(), num*group_num_, bbox_dim, 
            spatial_dim, bbox_bound_.cpu_data());
  }
  
  const Dtype* encoded_gt_data = gt_bbox_.cpu_diff();
  const Dtype* bbox_data = bottom[1]->cpu_data();
         
  for (int i = 0; i < num; ++i) {
    for (int nn = 0; nn < group_num_; nn++) {
      const int group_idx = i*group_num_+nn;
      for (int j = 0; j < spatial_dim; ++j) {
        const int base_index = group_idx*spatial_dim+j;
        const int label_value = static_cast<int>(gt_label_data[base_index]);
        const int keep_flag = static_cast<int>(keep_map_data[base_index]); 
        if ((has_ignore_label_ && label_value == ignore_label_) || 
                (label_value==0) || (keep_flag==0))  {
          continue;
        }
        for (int k = 0; k < coord_num_; k++) {
          const int idx = group_idx*bbox_dim + k*spatial_dim + j;
          Dtype diff = bbox_data[idx]-encoded_gt_data[idx];
          if (bbox_loss_type_ == "Euclidean_L2") {
            bbox_loss += 0.5*diff*diff;
            bbox_diff_data[idx] = diff; 
          } else if (bbox_loss_type_ == "Smooth_L1") {
            if (diff <= -1) { 
              bbox_loss += (std::abs(diff)-Dtype(0.5));
              bbox_diff_data[idx] = Dtype(-1);
            } else if (diff >= 1) {
              bbox_loss += (std::abs(diff)-Dtype(0.5));
              bbox_diff_data[idx] = Dtype(1);
            } else {
              bbox_loss += 0.5*diff*diff;
              bbox_diff_data[idx] = diff;
            }
          } else {
            LOG(FATAL) << "Unknown bbox loss type.";
          }
          bbox_count++;
        }
      }
    }
  }

  // normalize
  if (conf_count == 0) {
    conf_loss = Dtype(0);
  } else {
    conf_loss /= conf_count;
  }
  if (bbox_count == 0) {
    bbox_loss = Dtype(0);
  } else {
    bbox_loss /= bbox_count;
    // scale bbox gradient
    caffe_scal(bbox_diff_.count(), Dtype(1.0)/bbox_count, bbox_diff_data);
  }
  
  // combine the loss
  top[0]->mutable_cpu_data()[0] = conf_loss+lambda_*bbox_loss;
  top[0]->mutable_cpu_data()[1] = lambda_*bbox_loss;
  DLOG(INFO) << "conf_loss = "<<conf_loss<<", conf_count = "<<conf_count
             <<", bbox_loss = "<<bbox_loss<<", bbox_count = "<<bbox_count;
}

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  // gradient of class bottom
  if (propagate_down[0]) {
    const int num = bottom[0]->num() / group_num_;
    const Dtype* gt_label_data = gt_label_.cpu_data();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    Dtype* conf_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();    
    const Dtype* keep_map_data = keep_map_.cpu_data();
    const Dtype* weight_map_data = weight_map_.cpu_data();   
    const int conf_dim = bottom[0]->count() / bottom[0]->num();
    int conf_count = 0;    
  
    caffe_copy(prob_.count(), prob_data, conf_diff);
    for (int i = 0; i < num; ++i) {
      for (int nn = 0; nn < group_num_; nn++) {
        const int group_idx = i*group_num_+nn;
        for (int j = 0; j < spatial_dim; ++j) {
          const int base_index = group_idx*spatial_dim+j;
          int label_value = static_cast<int>(gt_label_data[base_index]);
          if (objectness_) {
            label_value = std::min(1,label_value);
          } 
          const int keep_flag = static_cast<int>(keep_map_data[base_index]);
          if ((has_ignore_label_ && label_value==ignore_label_) || (keep_flag==0)) {
            for (int c = 0; c < cls_num_; ++c) {
              conf_diff[group_idx * conf_dim + c * spatial_dim + j] = 0;
            }
          } else {
            conf_diff[group_idx * conf_dim + label_value * spatial_dim + j] -= 1;
            conf_count++;
          }
        }
      }
    }
    
    // weighting the class gradient
    for (int i = 0; i < num*group_num_; i++) {
      for (int j = 0; j < cls_num_; j++) {
        Dtype* tmp_diff = conf_diff+i*conf_dim+j*spatial_dim;
        caffe_mul(spatial_dim, tmp_diff, weight_map_data+i*spatial_dim, tmp_diff);
      }
    }
    
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (conf_count > 0) {
      caffe_scal(bottom[0]->count(), loss_weight / conf_count, conf_diff);
    } 
  }
  
  //gradient of coordinate bottom
  if (propagate_down[1]) {
    caffe_copy(bottom[1]->count(), bbox_diff_.cpu_data(), 
            bottom[1]->mutable_cpu_diff());
    // Scale gradient
    const Dtype loss_weight = lambda_*top[0]->cpu_diff()[0];
    caffe_scal(bottom[1]->count(), loss_weight, bottom[1]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(DetectionGroupLossLayer);
#endif

INSTANTIATE_CLASS(DetectionGroupLossLayer);
REGISTER_LAYER_CLASS(DetectionGroupLoss);

}  // namespace caffe
