// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <vector>

#include "caffe/layers/hard_mining_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void HardMiningLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // propagate_down=false for all bottoms
  if (this->layer_param_.propagate_down_size() == 0) {
    for (int k = 0; k < bottom.size(); k++) {
      this->layer_param_.add_propagate_down(false);
    }
  } else {
    for (int k = 0; k < bottom.size(); k++) {
      CHECK(!this->layer_param_.propagate_down(k));
    }
  }
  CHECK_EQ(bottom.size(),top.size()+1);
  num_roi_per_image_ = this->layer_param_.hard_mining_param().num_roi_per_image();
  num_image_ = this->layer_param_.hard_mining_param().num_image();
  CHECK_GT(num_roi_per_image_,0); CHECK_GT(num_image_,0);
}

template <typename Dtype>
void HardMiningLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // minimum bottom: rank_score and rois (rois is needed for image id)
  const int num_rois = bottom[0]->count();
  CHECK_EQ(num_rois,bottom[0]->num()); 
  batch_size_ = num_image_*num_roi_per_image_;
  batch_size_ = std::min(batch_size_,num_rois);
  //CHECK_GE(num_rois,batch_size_); 
  for (int k = 0; k < top.size(); k++) {
    CHECK_EQ(num_rois,bottom[k+1]->num());
    const int channels = bottom[k+1]->channels();
    const int height = bottom[k+1]->height();
    const int width = bottom[k+1]->width();
    top[k]->Reshape(batch_size_, channels, height, width);
  }
  // bottom[1]: rois (img_id, x1, y1, x2, y2)
  CHECK_EQ(bottom[1]->channels(),5);
}

template <typename Dtype>
void HardMiningLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* ranking_score = bottom[0]->cpu_data();
  const int num_rois = bottom[0]->num();
  const Dtype* rois_data = bottom[1]->cpu_data();
  const int rois_dim = bottom[1]->channels();
  vector<vector<pair<Dtype, int> > > image_rois(num_image_);
  for (int i = 0; i < num_rois; i++) {
    int image_id = rois_data[i*rois_dim];
    CHECK_GE(image_id, 0); CHECK_LT(image_id, num_image_);
    image_rois[image_id].push_back(std::make_pair(ranking_score[i],i));
  }
  // ranking per image
  for (int i = 0; i < num_image_; i++) {
    if (image_rois[i].size() == 0) {
      LOG(INFO) << "Couldn't find any rois";
      continue;
    }
    std::sort(image_rois[i].begin(), image_rois[i].end(), 
            std::greater<std::pair<Dtype, int> >());
  }

  // collecting hard instances per image
  vector<int> keep_index;
  vector<pair<Dtype, int> > left_rois;
  for (int i = 0; i < num_image_; i++) {
    if (image_rois[i].size() <= num_roi_per_image_) {
      for (int j = 0; j < image_rois[i].size(); j++) {
        keep_index.push_back(image_rois[i][j].second);
      }
    } else {
      for (int j = 0; j < num_roi_per_image_; j++) {
        keep_index.push_back(image_rois[i][j].second);
      }
      for (int j = num_roi_per_image_; j < image_rois[i].size(); j++) {
        left_rois.push_back(image_rois[i][j]);
      }
    }
  }
  
  // pickup some other rois if rois is not enough yet
  const int num_pickup = batch_size_-keep_index.size();
  if (num_pickup > 0) {
    std::partial_sort(left_rois.begin(), left_rois.begin()+num_pickup,
               left_rois.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < num_pickup; j++) {
      keep_index.push_back(left_rois[j].second);
    }
  }
  CHECK_EQ(batch_size_,keep_index.size());
  // re-order index
  std::sort(keep_index.begin(),keep_index.end());
  
  // copy bottoms to tops
  for (int k = 0; k < top.size(); k++) {
    const int dim = bottom[k+1]->count() / bottom[k+1]->num();
    CHECK_EQ(dim*batch_size_,top[k]->count());
    const Dtype* bottom_data = bottom[k+1]->cpu_data();
    Dtype* top_data = top[k]->mutable_cpu_data();
    for (int i = 0; i < batch_size_; i++) {
      int idx = keep_index[i]; CHECK_GT(bottom[k+1]->num(),idx);
      caffe_copy(dim, bottom_data+idx*dim, top_data+i*dim);
    }
  }
}

INSTANTIATE_CLASS(HardMiningLayer);
REGISTER_LAYER_CLASS(HardMining);

}  // namespace caffe
