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
__global__ void DetectionGroupSoftmaxForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const Dtype* keep_map, 
          const Dtype* weight_map, Dtype* loss, const int conf_dim, 
          const int spatial_dim, const bool objectness, const bool has_ignore_label, 
          const int ignore_label, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim; // group index
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
    if (objectness) {
      label_value = min(1,label_value);
    } 
    const int keep_flag = static_cast<int>(keep_map[index]);
    if ((has_ignore_label && label_value == ignore_label) || (keep_flag == 0)) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n*conf_dim+label_value*spatial_dim+s], 
                    Dtype(FLT_MIN))) * weight_map[index];
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void GroupBBoxEuclideanL2ForwardGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype* bbox_gt_data, const Dtype* label, 
          const Dtype* keep_map, const int bbox_dim, const int spatial_dim, 
          const int coord_num, const bool has_ignore_label, const int ignore_label, 
          Dtype* bbox_loss, Dtype* bbox_diff, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[index]);
    const int keep_flag = static_cast<int>(keep_map[index]);
    if ((has_ignore_label && label_value==ignore_label) || 
            (label_value==0) || (keep_flag==0)) {
      counts[index] = 0;
    } else {
      for (int k = 0; k < coord_num; k++) {
        const int idx = n*bbox_dim + k*spatial_dim + s;
        Dtype diff = bbox_data[idx]-bbox_gt_data[idx];
        bbox_loss[idx] = 0.5*diff*diff;
        bbox_diff[idx] = diff; 
      }
      counts[index] = coord_num;
    }
  }
}

template <typename Dtype>
__global__ void GroupBBoxSmoothL1ForwardGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype* bbox_gt_data, const Dtype* label, 
          const Dtype* keep_map, const int bbox_dim, const int spatial_dim, 
          const int coord_num, const bool has_ignore_label, const int ignore_label, 
          Dtype* bbox_loss, Dtype* bbox_diff, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[index]);
    const int keep_flag = static_cast<int>(keep_map[index]);
    if ((has_ignore_label && label_value==ignore_label) || 
            (label_value==0) || (keep_flag==0)) {
      counts[index] = 0;
    } else {
      for (int k = 0; k < coord_num; k++) {
        const int idx = n*bbox_dim + k*spatial_dim + s;
        Dtype diff = bbox_data[idx]-bbox_gt_data[idx];
        if (diff <= -1) { 
          bbox_loss[idx] = abs(diff)-Dtype(0.5);
          bbox_diff[idx] = Dtype(-1);
        } else if (diff >= 1) {
          bbox_loss[idx] = abs(diff)-Dtype(0.5);
          bbox_diff[idx] = Dtype(1);
        } else {
          bbox_loss[idx] = 0.5*diff*diff;
          bbox_diff[idx] = diff;
        } 
      }
      counts[index] = coord_num;
    }
  }
}

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_gt_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num() / group_num_;
  const int height = bottom[0]->height(); 
  const int width = bottom[0]->width();
  const int spatial_dim = height*width;
  const int conf_dim = bottom[0]->count() / bottom[0]->num();
  const int bbox_dim = bottom[1]->count() / bottom[1]->num();
  const int gt_dim = bottom[2]->count() / bottom[2]->num();
  const int nthreads = num * group_num_ * spatial_dim;  

  // extract gt data
  for (int i = 0; i < num; i++) {
    for (int nn = 0; nn < group_num_; nn++) {
      const int group_idx = i*group_num_+nn;
      caffe_copy(spatial_dim, bottom_gt_data+group_idx*gt_dim, 
              gt_label_.mutable_gpu_data()+group_idx*spatial_dim);
      caffe_copy(bbox_dim, bottom_gt_data+group_idx*gt_dim+spatial_dim, 
              gt_bbox_.mutable_gpu_data()+group_idx*bbox_dim);
      caffe_copy(spatial_dim, bottom_gt_data+group_idx*gt_dim+spatial_dim+bbox_dim, 
              gt_overlap_.mutable_gpu_data()+group_idx*spatial_dim);
    }
  }
  const Dtype* gt_label_data = gt_label_.gpu_data();
  const Dtype* gt_label_cpu = gt_label_.cpu_data();
  const Dtype* gt_bbox_data = gt_bbox_.gpu_data();
  const Dtype* gt_overlap_cpu = gt_overlap_.cpu_data();

  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_cpu = prob_.cpu_data();

  // assign negatives ranking scores
  Dtype* neg_ranking_cpu = keep_map_.mutable_cpu_diff();
  caffe_set(keep_map_.count(), Dtype(0), neg_ranking_cpu);
  if ((neg_mining_type_ == "bootstrap") || (neg_mining_type_ == "mixture")) {
    for (int nn = 0; nn < group_num_; nn++) {
      if (neg_ranking_type_ == "conf_prob") {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < spatial_dim; j++) {
            neg_ranking_cpu[(i*group_num_+nn)*spatial_dim+j] 
                    = 1-prob_cpu[(i*group_num_+nn)*conf_dim+j];
          }
        } 
      } else if (neg_ranking_type_ == "conf_loss") {
        for (int i = 0; i < num; ++i) {
          const int group_idx = i*group_num_+nn;
          for (int j = 0; j < spatial_dim; j++) {
            int label_value = static_cast<int>(gt_label_cpu[group_idx*spatial_dim+j]);
            if (objectness_) {
              label_value = std::min(1,label_value);
            }
            CHECK_GE(label_value, 0);
            CHECK_LT(label_value, prob_.channels());
            neg_ranking_cpu[group_idx*spatial_dim+j] = -log(std::max(
                    prob_cpu[group_idx*conf_dim+label_value*spatial_dim+j],Dtype(FLT_MIN)));
          }
        }
      } else {
        LOG(FATAL) << "Unknown negative ranking type.";
      }
    }
  }

  //build the keep map
  Dtype* keep_map_cpu = keep_map_.mutable_cpu_data();
  caffe_set(keep_map_.count(), Dtype(0), keep_map_cpu);
  int total_neg_num = 0, total_pos_num = 0, total_sample_num = 0;

  for (int i = 0; i < num; ++i) {
    int num_pos, num_neg, sample_pos_num, sample_neg_num;
    // collecting potential positives and negatives
    std::vector<int> pos_keep_idxs, neg_keep_idxs;
    std::vector<std::pair<Dtype, int> > neg_score_idx_pairs;
    for (int nn = 0; nn < group_num_; nn++) {
      for (int j = 0; j < spatial_dim; j++) {
        const int base_index = (i*group_num_+nn)*spatial_dim+j;
        const int label_value = static_cast<int>(gt_label_cpu[base_index]);
        const Dtype overlap = gt_overlap_cpu[base_index];
        if (label_value > 0) {
          pos_keep_idxs.push_back(nn*spatial_dim+j);
        } else if ((label_value==0) && (overlap<bg_threshold_)) {
          neg_score_idx_pairs.push_back(std::make_pair(neg_ranking_cpu[base_index], 
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
      keep_map_cpu[i*group_num_*spatial_dim+keep_idx] = Dtype(1);
    }
    for (int j = 0; j < sample_neg_num; j++) {
      const int keep_idx = neg_keep_idxs[j];
      keep_map_cpu[i*group_num_*spatial_dim+keep_idx] = Dtype(1);
    } 
  }
  int total_sample_num_check = caffe_cpu_asum(keep_map_.count(),keep_map_cpu);
  CHECK_EQ(total_sample_num, total_sample_num_check);
  DLOG(INFO)<<"total positive = "<<total_pos_num <<", total negative = "
          <<total_neg_num<<", keep = "<<total_sample_num;

  // setup weight map
  Dtype* weight_map_cpu = weight_map_.mutable_cpu_data();
  caffe_set(weight_map_.count(), Dtype(1), weight_map_cpu);
  Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
  if (pos_neg_weighted_) {
    for (int i = 0; i < weight_map_.count(); i++) {
      const int label_value = static_cast<int>(gt_label_cpu[i]);
      if (keep_map_cpu[i] == 0) continue;
      if (label_value > 0) {
        pos_weight_sum += weight_map_cpu[i]; 
      } else {
        neg_weight_sum += weight_map_cpu[i]; 
      }
    }
    Dtype fg_weight = Dtype(1)/(1+bg_multiple_);
    for (int i = 0; i < weight_map_.count(); i++) {
      const int label_value = static_cast<int>(gt_label_cpu[i]);
      if (label_value > 0) {
        if (pos_weight_sum != 0) {
          weight_map_cpu[i] *= (fg_weight*total_sample_num/pos_weight_sum); 
        }
      } else {
        if (neg_weight_sum != 0) {
          weight_map_cpu[i] *= ((1-fg_weight)*total_sample_num/neg_weight_sum); 
        }
      }
    }
  }

  Dtype* conf_loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* conf_counts = prob_.mutable_gpu_diff();
  const Dtype* keep_map = keep_map_.gpu_data();
  const Dtype* weight_map = weight_map_.gpu_data();
  DetectionGroupSoftmaxForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_.gpu_data(), gt_label_data, 
      keep_map, weight_map, conf_loss_data, conf_dim, spatial_dim, 
      objectness_, has_ignore_label_, ignore_label_, conf_counts);

  // encode gt bbox
  EncodeGroupBBoxesGPU(nthreads, gt_bbox_.gpu_data(), gt_label_data, group_num_, 
          width, height, bbox_mean_.gpu_data(), bbox_std_.gpu_data(), field_ws_.gpu_data(), 
          field_hs_.gpu_data(), stride_, gt_bbox_.mutable_gpu_diff());
  
  // bound predictions
  if (do_bound_bbox_) {
    BoundBBoxPredsGPU(nthreads, bottom[1]->mutable_gpu_data(), bbox_dim, 
            spatial_dim, bbox_bound_.gpu_data());
  }

  // the forward pass computes bbox loss 
  Dtype* bbox_loss_data = bottom[1]->mutable_gpu_diff();
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bbox_loss_data);
  Dtype* bbox_diff_data = bbox_diff_.mutable_gpu_data();
  caffe_gpu_set(bbox_diff_.count(), Dtype(0), bbox_diff_data);
  Dtype* bbox_counts = weight_map_.mutable_gpu_diff();
  caffe_gpu_set(weight_map_.count(), Dtype(0), bbox_counts);
  const Dtype* encoded_gt_data = gt_bbox_.gpu_diff();
  const Dtype* bbox_data = bottom[1]->gpu_data();

  if (bbox_loss_type_ == "Euclidean_L2") {
    GroupBBoxEuclideanL2ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, encoded_gt_data, 
        gt_label_data, keep_map, bbox_dim, spatial_dim, coord_num_, has_ignore_label_, 
        ignore_label_, bbox_loss_data, bbox_diff_data, bbox_counts);
  } else if (bbox_loss_type_ == "Smooth_L1") {
    GroupBBoxSmoothL1ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, encoded_gt_data, 
        gt_label_data, keep_map, bbox_dim, spatial_dim, coord_num_, has_ignore_label_, 
        ignore_label_, bbox_loss_data, bbox_diff_data, bbox_counts);
  } else {
    LOG(FATAL) << "Unknown bbox loss type.";
  }

  Dtype conf_loss, bbox_loss;
  Dtype conf_count, bbox_count;
  caffe_gpu_asum(nthreads, conf_loss_data, &conf_loss);
  caffe_gpu_asum(nthreads, conf_counts, &conf_count); 
  caffe_gpu_asum(nthreads*coord_num_, bbox_loss_data, &bbox_loss);
  caffe_gpu_asum(nthreads, bbox_counts, &bbox_count);
  if (conf_count == 0) {
    conf_loss = 0;
  } else {
    conf_loss /= conf_count;
  }
  if (bbox_count == 0) {
    bbox_loss = 0;
  } else {
    bbox_loss /= bbox_count;
    // scale bbox gradient
    caffe_gpu_scal(bbox_diff_.count(), Dtype(1.0)/bbox_count, bbox_diff_data);
  }
  
  // combine the loss
  top[0]->mutable_cpu_data()[0] = conf_loss+lambda_*bbox_loss;
  top[0]->mutable_cpu_data()[1] = lambda_*bbox_loss;
  DLOG(INFO) << "conf_loss = "<<conf_loss<<", conf_count = "<<conf_count
             <<", bbox_loss = "<<bbox_loss<<", bbox_count = "<<bbox_count;
}

template <typename Dtype>
__global__ void DetectionGroupSoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* label, const Dtype* keep_map, Dtype* conf_diff, 
          const int conf_dim, const int spatial_dim, const int channels, 
          const bool objectness, const bool has_ignore_label, 
          const int ignore_label, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
    if (objectness) {
      label_value = min(1,label_value);
    } 
    const int keep_flag = static_cast<int>(keep_map[index]);
    if ((has_ignore_label && label_value==ignore_label) || (keep_flag==0)) {
      for (int c = 0; c < channels; ++c) {
        conf_diff[n * conf_dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      conf_diff[n * conf_dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void DetectionGroupLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    const int num = bottom[0]->num() / group_num_;
    const Dtype* gt_label_data = gt_label_.gpu_data();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    Dtype* conf_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* keep_map = keep_map_.gpu_data();
    const Dtype* weight_map = weight_map_.gpu_data();
    const int conf_dim = bottom[0]->count() / bottom[0]->num();
    const int nthreads = num * group_num_ * spatial_dim;

    caffe_copy(prob_.count(), prob_data, conf_diff);
    Dtype* conf_counts = prob_.mutable_gpu_diff();
    caffe_gpu_set(prob_.count(),Dtype(0),conf_counts);
    DetectionGroupSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, gt_label_data, keep_map, conf_diff, 
        conf_dim, spatial_dim, cls_num_, objectness_, has_ignore_label_, 
        ignore_label_, conf_counts);

    // weihgting the class gradient
    for (int i = 0; i < num*group_num_; i++) {
      for (int j = 0; j < cls_num_; j++) {
        Dtype* tmp_diff = conf_diff+i*conf_dim+j*spatial_dim;
        caffe_gpu_mul(spatial_dim, tmp_diff, weight_map+i*spatial_dim, tmp_diff);
      }
    }

    // Scale gradient
    Dtype conf_count;
    caffe_gpu_asum(nthreads, conf_counts, &conf_count);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (conf_count > 0) {
      caffe_gpu_scal(bottom[0]->count(), loss_weight / conf_count, conf_diff);
    } 
  }

  //gradient of coordinate bottom
  if (propagate_down[1]) {
    caffe_copy(bottom[1]->count(), bbox_diff_.gpu_data(), 
            bottom[1]->mutable_gpu_diff());
    // Scale gradient
    const Dtype loss_weight = lambda_*top[0]->cpu_diff()[0];
    caffe_gpu_scal(bottom[1]->count(), loss_weight, bottom[1]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionGroupLossLayer);

}  // namespace caffe
