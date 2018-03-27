// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/detection_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"

// caffe.proto > LayerParameter > DetectionDataParameter

namespace caffe {

template <typename Dtype>
DetectionDataLayer<Dtype>::~DetectionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_gts
  //    label ignore difficult x1 y1 x2 y2
  const DetectionDataParameter& detection_data_param = 
          this->layer_param_.detection_data_param();
  
  LOG(INFO) << "Window data layer:" << std::endl
      << "  batch size: "
      << detection_data_param.batch_size() << std::endl
      << "  cache_images: "
      << detection_data_param.cache_images() << std::endl
      << "  root_folder: "
      << detection_data_param.root_folder();

  
  cache_images_ = detection_data_param.cache_images();
  string root_folder = detection_data_param.root_folder();

  // reset the random generator seed
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  std::ifstream infile(detection_data_param.source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << detection_data_param.source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels, img_height, img_width;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0]; img_height = image_size[1]; img_width = image_size[2];
    image_database_.push_back(std::make_pair(image_path, image_size));
    image_list_.push_back(image_index);
    if (img_height>=img_width) {
      longer_height_list_.push_back(image_index);
    } else {
      longer_width_list_.push_back(image_index);
    }

    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
    vector<BBox> windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, ignore, difficult, x1, y1, x2, y2;
      infile >> label >> ignore >> difficult >> x1 >> y1 >> x2 >> y2;
      BBox window = InitBBox(x1,y1,x2,y2);
      window.label = label;
      window.ignore = ignore;
      window.difficult = difficult;
      CHECK_GT(label, 0);
      windows.push_back(window);
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }
    windows_.push_back(windows);

    int num_roni_windows;
    vector<BBox> roni_windows;
    infile >> num_roni_windows;
    for (int i = 0; i < num_roni_windows; ++i) {
      int x1, y1, x2, y2;      
      infile >> x1 >> y1 >> x2 >> y2;
      BBox roni_window = InitBBox(x1,y1,x2,y2);
      roni_windows.push_back(roni_window);
    }
    roni_windows_.push_back(roni_windows);

    if (image_index % 1000 == 0) {
      LOG(INFO) << "num: " << image_index << " " << image_path << " "
          << image_size[0] << " " << image_size[1] << " " << image_size[2] << " "
          << "windows to process: " << num_windows
          << ", RONI windows: "<< num_roni_windows;
    }
  } while (infile >> hashtag >> image_index);

  CHECK_EQ(image_list_.size(),longer_height_list_.size()+longer_width_list_.size());
  LOG(INFO) << "Number of images: " << image_index+1;
  CHECK_EQ(windows_.size(),image_database_.size());
  CHECK_EQ(windows_.size(),roni_windows_.size());
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }
  
  // randomly shuffle data
  mix_group_size_ = detection_data_param.mix_group_size();
  if (detection_data_param.has_shuffle()) {
    if (detection_data_param.shuffle() == "random_aspect") {
      LOG(INFO) << "Random aspect shuffling data";
      ShuffleAspectGroupList();
    } else if (detection_data_param.shuffle() == "random_mix_aspect") {
      LOG(INFO) << "Random mixture shuffling data";
      ShuffleMixAspectGroupList();
    } else if (detection_data_param.shuffle() == "random") {
      LOG(INFO) << "Random shuffling data";
      ShuffleList();
    } else {
      CHECK(false) << "Unknown shuffling Function!";
    }
  }
  
  list_id_ = 0;
  
  // decide the template size
  int template_height, template_width;
  if (detection_data_param.has_resize_width() && 
          detection_data_param.has_resize_height()) {
    CHECK(!(detection_data_param.has_short_size() || 
            detection_data_param.has_long_size())) << "No need for short and long size!";
    CHECK(!detection_data_param.has_resize_ratio()) << "No need for resize ratio!";
    template_width = detection_data_param.resize_width();
    template_height = detection_data_param.resize_height();
  } else if (detection_data_param.has_resize_ratio()) {
    CHECK(!(detection_data_param.has_resize_width() || 
            detection_data_param.has_resize_height())) << "No need for resize width and height!";
    CHECK(!(detection_data_param.has_short_size() || 
            detection_data_param.has_long_size())) << "No need for short and long size!";
    float resize_ratio = detection_data_param.resize_ratio();
    template_width = round(img_width*resize_ratio); 
    template_height = round(img_height*resize_ratio);
  } else if (detection_data_param.has_short_size() && 
          detection_data_param.has_long_size()) {
    CHECK_GT(detection_data_param.long_size(),detection_data_param.short_size());
    CHECK(!(detection_data_param.has_resize_width() || 
            detection_data_param.has_resize_height())) << "No need for resize width and height!";
    CHECK(!detection_data_param.has_resize_ratio()) << "No need for resize ratio!";
    template_width = detection_data_param.short_size();
    template_height = detection_data_param.long_size();
  } else {
    template_width = img_width; template_height = img_height;
  }
  
  if (detection_data_param.has_crop_width()
      && detection_data_param.has_crop_height()) {
    CHECK(!(detection_data_param.has_crop_short_size() || 
            detection_data_param.has_crop_long_size())) 
            << "No need for crop short and long size!";
    int crop_width = detection_data_param.crop_width();
    int crop_height = detection_data_param.crop_height();
    CHECK_GT(crop_width,0); CHECK_GT(crop_height,0);
    template_width = crop_width; template_height = crop_height;
  } else if (detection_data_param.has_crop_short_size() && 
          detection_data_param.has_crop_long_size()) {
    CHECK(!(detection_data_param.has_crop_width() || 
            detection_data_param.has_crop_height())) 
            << "No need crop width and height!";
    CHECK_GT(detection_data_param.crop_long_size(),detection_data_param.crop_short_size());
    template_width = detection_data_param.crop_short_size();
    template_height = detection_data_param.crop_long_size();
  }
  
  // image reshape
  const int batch_size = detection_data_param.batch_size();
  top[0]->Reshape(batch_size, channels, template_height, template_width);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(batch_size, channels, template_height, template_width);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
  // label reshape (label_id, x, y, w, h, overlap)
  label_channel_ = 2+detection_data_param.coord_num();
  label_blob_num_ = detection_data_param.stride_size();
  //CHECK_GE(label_blob_num_, 1);
  CHECK_EQ(label_blob_num_, detection_data_param.field_h_size());
  CHECK_EQ(label_blob_num_, detection_data_param.field_w_size());
  for (int nn = 0; nn < label_blob_num_; nn++) {
    strides_.push_back(detection_data_param.stride(nn));
    field_ws_.push_back(detection_data_param.field_w(nn));
    field_hs_.push_back(detection_data_param.field_h(nn));
    int label_height = round(float(template_height)/strides_[nn]); 
    int label_width = round(float(template_width)/strides_[nn]);
    top[nn+1]->Reshape(batch_size, label_channel_, label_height, label_width);
    
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(batch_size, label_channel_, label_height, label_width);
      this->prefetch_[i]->labels_.push_back(label_blob_pointer);
    }
    LOG(INFO) << "output label size "<<nn<<" : " << top[nn+1]->num() << ","
        << top[nn+1]->channels() << "," << top[nn+1]->height() << ","
        << top[nn+1]->width();
  }

  // setup for output gt boxes
  output_gt_boxes_ = detection_data_param.output_gt_boxes();
  if (output_gt_boxes_) {
    //dummy reshape
    const int gt_dim = 8;
    top[label_blob_num_+1]->Reshape(1, gt_dim, 1, 1);    
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(1, gt_dim, 1, 1);
      this->prefetch_[i]->labels_.push_back(label_blob_pointer);
    }
    LOG(INFO) << "output gt boxes size: " << top[label_blob_num_+1]->num() << ","
        << top[label_blob_num_+1]->channels() << "," << top[label_blob_num_+1]->height() 
        << "," << top[label_blob_num_+1]->width();
  }

  // data mean
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_values_) {
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
unsigned int DetectionDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::ShuffleList() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_list_.begin(), image_list_.end(), prefetch_rng);
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::ShuffleAspectGroupList() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(longer_height_list_.begin(), longer_height_list_.end(), prefetch_rng);
  prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(longer_width_list_.begin(), longer_width_list_.end(), prefetch_rng);
  vector<vector<int> > mix_list(2); 
  int idx = 0;
  if (PrefetchRand() % 2) {
    mix_list[0] = longer_height_list_; mix_list[1] = longer_width_list_; 
  } else {
    mix_list[0] = longer_width_list_; mix_list[1] = longer_height_list_; 
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < mix_list[i].size(); j++) {
      image_list_[idx++] = mix_list[i][j];
    }
  }
  CHECK_EQ(idx,image_list_.size());
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::ShuffleMixAspectGroupList() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  image_list_.clear();
  shuffle(longer_height_list_.begin(), longer_height_list_.end(), prefetch_rng);
  prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(longer_width_list_.begin(), longer_width_list_.end(), prefetch_rng);
  vector<vector<int> > mix_list(2); 
  if (PrefetchRand() % 2) {
    mix_list[0] = longer_height_list_; mix_list[1] = longer_width_list_; 
  } else {
    mix_list[0] = longer_width_list_; mix_list[1] = longer_height_list_; 
  }
  int num_image = longer_height_list_.size()+longer_width_list_.size();
  vector<int> mix_counts(2,0); int image_idx = 0;
  while ((mix_counts[0]<mix_list[0].size()) && (mix_counts[1]<mix_list[1].size())) {
    const int group_id = image_idx / mix_group_size_;
    const int mix_id = group_id % 2;
    const int list_id = mix_list[mix_id][mix_counts[mix_id]];
    image_list_.push_back(list_id); 
    image_idx++; mix_counts[mix_id]++;
  }
  if (mix_counts[0] < mix_list[0].size()) {
    CHECK_EQ(mix_counts[1],mix_list[1].size());
    for (int i = mix_counts[0]; i < mix_list[0].size(); i++) {
      image_list_.push_back(mix_list[0][i]);
    }
  } else if (mix_counts[1] < mix_list[1].size()) {
    CHECK_EQ(mix_counts[0],mix_list[0].size());
    for (int i = mix_counts[1]; i < mix_list[1].size(); i++) {
      image_list_.push_back(mix_list[1][i]);
    }
  }
  CHECK_EQ(image_list_.size(),num_image);
}

// Thread fetching the data
template <typename Dtype>
void DetectionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  double label_time = 0;
  CPUTimer timer;
  Dtype* top_data = NULL;
  int template_width = batch->data_.width();
  int template_height = batch->data_.height(); 

  const DetectionDataParameter& detection_data_param = 
          this->layer_param_.detection_data_param();
  const Dtype scale = this->transform_param_.scale();
  const int batch_size = detection_data_param.batch_size();
  const bool mirror = this->transform_param_.mirror();
  const float fg_threshold = detection_data_param.fg_threshold();
  CHECK_GT(fg_threshold,0);
  const float ignore_fg_threshold = detection_data_param.ignore_fg_threshold();
  const float min_gt_width = detection_data_param.min_gt_width();
  const float min_gt_height = detection_data_param.min_gt_height();
  if (detection_data_param.has_min_scale() && detection_data_param.has_max_scale()) {
    CHECK(!(detection_data_param.has_low_scale_ratio() || 
            detection_data_param.has_high_scale_ratio())) 
            << "No need low and high scale ratio!";
  }
  if (detection_data_param.has_low_scale_ratio() && 
          detection_data_param.has_high_scale_ratio()) {
    CHECK(!(detection_data_param.has_min_scale() || 
            detection_data_param.has_max_scale())) 
            << "No need low and high scale ratio!";
  }
  const bool multiscale_flag = (detection_data_param.has_min_scale() 
                        && detection_data_param.has_max_scale()) 
                        || (detection_data_param.has_low_scale_ratio() 
                        && detection_data_param.has_high_scale_ratio());
  const float multiscale_prob = detection_data_param.multiscale_prob();
  const bool do_wh_aspect = detection_data_param.has_min_whaspect()
                        && detection_data_param.has_max_whaspect();
  const bool need_reshape = !(
          (detection_data_param.has_crop_width() && detection_data_param.has_crop_height()) || 
          (detection_data_param.has_resize_width() && detection_data_param.has_resize_height()));
  
  float template_ratio = 1.0;
  if (detection_data_param.template_ratio_size() > 0) {
    int template_ratio_size = detection_data_param.template_ratio_size();
    int size_id = PrefetchRand() % template_ratio_size;
    template_ratio = detection_data_param.template_ratio(size_id);
  }
  // label
  vector<Dtype*> top_labels(label_blob_num_); 
  vector<int> label_spatial_dims(label_blob_num_);
  
  // gt boxes
  vector<vector<Dtype> > gt_boxes;
  
  int image_database_size = image_database_.size();
  CHECK_EQ(image_list_.size(),image_database_size);
  CHECK_EQ(image_database_size,windows_.size());
  CHECK_EQ(image_database_size,roni_windows_.size());
  // sample from bg set then fg set
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      // sample a window
      timer.Start();
      CHECK_GT(image_database_size, list_id_);
      const unsigned int image_window_index = image_list_[list_id_];
      vector<BBox> windows = windows_[image_window_index];
      vector<BBox> roni_windows = roni_windows_[image_window_index];
      
      // load the image containing the window
      pair<std::string, vector<int> > image = image_database_[image_window_index];

      cv::Mat cv_img;
      if (this->cache_images_) {
        pair<std::string, Datum> image_cached = image_database_cache_[image_window_index];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
      }
      if (detection_data_param.has_distort_param()) {
        cv_img = ApplyDistort(cv_img, detection_data_param.distort_param());
      }
      
      int img_height = cv_img.rows, img_width = cv_img.cols;
      read_time += timer.MicroSeconds();
      timer.Start();
           
      // horizontal flip at random
      bool do_mirror = mirror && PrefetchRand() % 2;
      if (do_mirror) {
        cv::flip(cv_img, cv_img, 1);
        FlipBBoxes(windows,img_width);
        FlipBBoxes(roni_windows,img_width);
      }
      
      // resize image if needed
      int resize_width, resize_height;
      if (detection_data_param.has_resize_width()
          && detection_data_param.has_resize_height()) {
        resize_width = detection_data_param.resize_width();
        resize_height = detection_data_param.resize_height();  
      } else if (detection_data_param.has_resize_ratio()) {
        float resize_ratio = detection_data_param.resize_ratio();
        resize_width = round(img_width*resize_ratio); 
        resize_height = round(img_height*resize_ratio);
      } else if (detection_data_param.has_short_size() &&
          detection_data_param.has_long_size()) {
        int short_size = detection_data_param.short_size();
        short_size = round(short_size*template_ratio/32)*32;
        int img_short_size = std::min(img_width,img_height);
        float resize_scale = float(short_size)/float(img_short_size);
        resize_width = round(img_width*resize_scale);
        resize_height = round(img_height*resize_scale);
      } else {
        resize_width = img_width, resize_height = img_height;
      }
      if (resize_width != img_width || resize_height != img_height) {
        float width_factor = float(resize_width)/img_width;
        float height_factor = float(resize_height)/img_height;
        cv::Size cv_resize_size;
        cv_resize_size.width = resize_width; cv_resize_size.height = resize_height;
        cv::resize(cv_img, cv_img, cv_resize_size, 0, 0, cv::INTER_LINEAR);
        // resize bounding boxes
        AffineBBoxes(windows,width_factor,height_factor,0,0);
        AffineBBoxes(roni_windows,width_factor,height_factor,0,0);
      }
      img_height = cv_img.rows, img_width = cv_img.cols;
      
      // decide blob reshape if needed
      if (item_id == 0 && need_reshape) {
        if (detection_data_param.has_crop_short_size() && 
                detection_data_param.has_crop_long_size()) {
          if (img_height >= img_width) {
            template_width = detection_data_param.crop_short_size(); 
            template_height = detection_data_param.crop_long_size();
          } else {
            template_width = detection_data_param.crop_long_size(); 
            template_height = detection_data_param.crop_short_size();
          }
        } else if (detection_data_param.has_short_size() && 
                detection_data_param.has_long_size()) {
          if (img_height >= img_width) {
            template_width = detection_data_param.short_size(); 
            template_height = detection_data_param.long_size();
          } else {
            template_width = detection_data_param.long_size(); 
            template_height = detection_data_param.short_size();
          }
          template_width = round(template_width*template_ratio/32)*32;
          template_height = round(template_height*template_ratio/32)*32;
        } else {
          template_width = img_width; template_height = img_height;
        }
      }

      int src_offset_x=0, src_offset_y=0, dst_offset_x=0, dst_offset_y=0;
      int copy_width = template_width, copy_height = template_height;
      float width_rescale_factor  = 1, height_rescale_factor = 1;
      int sel_id=-1; float sel_center_x, sel_center_y;
      if (windows.size() != 0) {
        sel_id = PrefetchRand() % windows.size();
        sel_center_x = (windows[sel_id].xmin+windows[sel_id].xmax)/2.0;
        sel_center_y = (windows[sel_id].ymin+windows[sel_id].ymax)/2.0;
      } else {
        if (template_width > img_width) {
          sel_center_x = img_width/2.0;
        } else {
          sel_center_x = PrefetchRand() % (img_width-template_width+1) + template_width/2.0;
        }
        if (template_height > img_height) {
          sel_center_y = img_height/2.0;
        } else {
          sel_center_y = PrefetchRand() % (img_height-template_height+1) + template_height/2.0;
        }
      }

      int multiscale_action;
      caffe_rng_bernoulli(1, multiscale_prob, &multiscale_action);
      bool do_multiscale = multiscale_action && multiscale_flag;
      if (do_multiscale) {
        float low_scale_ratio, high_scale_ratio;
        if (detection_data_param.has_low_scale_ratio() && 
                detection_data_param.has_high_scale_ratio()) {
          low_scale_ratio = log2(detection_data_param.low_scale_ratio());
          high_scale_ratio = log2(detection_data_param.high_scale_ratio());
        } else if (detection_data_param.has_min_scale() && 
                detection_data_param.has_max_scale()) {
          float min_scale = detection_data_param.min_scale();
          float max_scale = detection_data_param.max_scale();
          if (windows.size()!=0) {
            float obj_width = windows[sel_id].xmax-windows[sel_id].xmin+1;
            float obj_height = windows[sel_id].ymax-windows[sel_id].ymin+1;
            float obj_scale = sqrt(obj_width*obj_height);
            low_scale_ratio = log2(min_scale/obj_scale);
            high_scale_ratio = log2(max_scale/obj_scale);
          } else {
            low_scale_ratio = 1.0; high_scale_ratio = 1.0;
          }
        } else {
          LOG(FATAL)<< "unknown multi-scale strategy!";
        }
        CHECK_GE(high_scale_ratio,low_scale_ratio);
        float random_log_scale;
        caffe_rng_uniform(1, low_scale_ratio, high_scale_ratio, &random_log_scale);
        float rescale_factor = pow(2,random_log_scale);
        width_rescale_factor = rescale_factor; height_rescale_factor = rescale_factor;
        if (do_wh_aspect) {
          float log_min_whaspect = log2(detection_data_param.min_whaspect());
          float log_max_whaspect = log2(detection_data_param.max_whaspect());
          CHECK_GT(log_max_whaspect,log_min_whaspect);
          float random_log_aspect;
          caffe_rng_uniform(1, log_min_whaspect, log_max_whaspect, &random_log_aspect);
          float random_aspect_ratio = pow(2,random_log_aspect);
          width_rescale_factor *= sqrt(random_aspect_ratio);
          height_rescale_factor /= sqrt(random_aspect_ratio);
        }
        DLOG(INFO)<<", rescale_factor: " <<rescale_factor<<", final_aspect: "
                <<width_rescale_factor/height_rescale_factor;
      }

      int rescale_height = round(img_height*height_rescale_factor);
      int rescale_width = round(img_width*width_rescale_factor);
      if (width_rescale_factor != 1 || height_rescale_factor != 1) {
        // if upsampling is too large, crop the image first, then upsample
        if (width_rescale_factor>1.5 || height_rescale_factor>1.5) {
          int crop_w = round(1.2*img_width/width_rescale_factor);
          int crop_h = round(1.2*img_height/height_rescale_factor);
          crop_w = std::min(crop_w,img_width);
          crop_h = std::min(crop_h,img_height);
          int crop_x1 = round(sel_center_x-crop_w*0.5); 
          int crop_y1 = round(sel_center_y-crop_h*0.5); 
          crop_x1 = std::max(crop_x1,0); crop_y1 = std::max(crop_y1,0);
          int diff_x = std::max(crop_x1+crop_w-img_width,0); crop_x1 -= diff_x;
          int diff_y = std::max(crop_y1+crop_h-img_height,0); crop_y1 -= diff_y;
          CHECK_GE(crop_x1,0); CHECK_GE(crop_y1,0);
          // crop image
          cv::Rect roi(crop_x1, crop_y1, crop_w, crop_h);
          cv_img = cv_img(roi);
          //shift center coordinates
          sel_center_x -= crop_x1; sel_center_y -= crop_y1;
          //shift bounding boxes
          AffineBBoxes(windows,1,1,-crop_x1,-crop_y1);
          AffineBBoxes(roni_windows,1,1,-crop_x1,-crop_y1);
          rescale_width = round(cv_img.cols*width_rescale_factor); 
          rescale_height = round(cv_img.rows*height_rescale_factor);
        }
        cv::Size cv_rescale_size;
        cv_rescale_size.width = rescale_width; cv_rescale_size.height = rescale_height;
        cv::resize(cv_img, cv_img, cv_rescale_size, 0, 0, cv::INTER_LINEAR);
        img_height = cv_img.rows, img_width = cv_img.cols;
      }
        
      // resize bounding boxes
      AffineBBoxes(windows,width_rescale_factor,height_rescale_factor,0,0);
      AffineBBoxes(roni_windows,width_rescale_factor,height_rescale_factor,0,0);

      int noise_x = PrefetchRand() % 20 - 10, noise_y = PrefetchRand() % 20 - 10;   
      if (rescale_width < template_width) {
        dst_offset_x = 0; copy_width = rescale_width; 
        src_offset_x = round((template_width-rescale_width)/2.0) + noise_x;
        src_offset_x = std::max(0,src_offset_x);
        src_offset_x = std::min(template_width-rescale_width,src_offset_x);
      } else if (rescale_width > template_width) {
        src_offset_x = 0; copy_width = template_width;
        int center_x = round(sel_center_x*width_rescale_factor)+noise_x;
        dst_offset_x = center_x-round(template_width/2.0);
        dst_offset_x = std::max(0,dst_offset_x);
        dst_offset_x = std::min(rescale_width-template_width,dst_offset_x);
      } else {
        src_offset_x = 0; dst_offset_x = 0; copy_width = template_width; 
      }
     
      if (rescale_height < template_height) {
        dst_offset_y = 0; copy_height = rescale_height; 
        src_offset_y = round((template_height-rescale_height)/2.0) + noise_y;
        src_offset_y = std::max(0,src_offset_y);
        src_offset_y = std::min(template_height-rescale_height,src_offset_y);
      } else if (rescale_height > template_height) {
        src_offset_y = 0; copy_height = template_height;
        int center_y = round(sel_center_y*height_rescale_factor)+noise_y;
        dst_offset_y = center_y-round(template_height/2.0);
        dst_offset_y = std::max(0,dst_offset_y);
        dst_offset_y = std::min(rescale_height-template_height,dst_offset_y);
      } else {
        src_offset_y = 0; dst_offset_y = 0; copy_height = template_height;
      }
        
      // shift bounding boxes
      AffineBBoxes(windows,1,1,src_offset_x-dst_offset_x,src_offset_y-dst_offset_y);
      AffineBBoxes(roni_windows,1,1,src_offset_x-dst_offset_x,src_offset_y-dst_offset_y);

      // blob reshape before copy
      if (item_id==0) {
        // data
        int channels = batch->data_.channels();
        batch->data_.Reshape(batch_size, channels, template_height, template_width);
        top_data = batch->data_.mutable_cpu_data();
        // zero out data batch
        caffe_set(batch->data_.count(), Dtype(0), top_data);
        // label
        for (int nn = 0; nn < label_blob_num_; nn++) {
          int label_height = round(float(template_height)/strides_[nn]); 
          int label_width = round(float(template_width)/strides_[nn]);
          batch->labels_[nn]->Reshape(batch_size, label_channel_, label_height, label_width);
          top_labels[nn] = batch->labels_[nn]->mutable_cpu_data();
          int label_count = batch->labels_[nn]->count();
          label_spatial_dims[nn] = batch->labels_[nn]->width()*batch->labels_[nn]->height();
          // zero out label batch
          caffe_set(label_count, Dtype(0), top_labels[nn]);
        }  
      }

      // copy the original image into top_data
      const int channels = cv_img.channels();     
      CHECK_LE(copy_width,template_width); CHECK_LE(copy_height,template_height);
      CHECK_LE(copy_width,img_width); CHECK_LE(copy_height,img_height);
      for (int h = src_offset_y; h < src_offset_y+copy_height; ++h) {
        const uchar* ptr = cv_img.ptr<uchar>(h-src_offset_y+dst_offset_y);
        for (int w = src_offset_x; w < src_offset_x+copy_width; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id*channels+c)*template_height+h)*template_width+w;
            int img_index = (w-src_offset_x+dst_offset_x) * channels + c;
            Dtype pixel = static_cast<Dtype>(ptr[img_index]);
            if (this->has_mean_values_) {
              top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
            } else {
              top_data[top_index] = pixel * scale;
            }
          }
        }
      }
      trans_time += timer.MicroSeconds();
      timer.Start();
      
      // get window label
      CHECK_EQ(label_channel_,6);
      vector<int> gt_match_counts(windows.size());
      vector<int> max_gt_blobid(windows.size());
      vector<Dtype> max_gt_iou(windows.size());
      for (int ww = 0; ww < windows.size(); ++ww) {
        Dtype bbxc = (windows[ww].xmin+windows[ww].xmax)/2.0;
        Dtype bbyc = (windows[ww].ymin+windows[ww].ymax)/2.0;
        Dtype bbw = windows[ww].xmax-windows[ww].xmin+1;
        Dtype bbh = windows[ww].ymax-windows[ww].ymin+1;
        //ignore gt bboxes whose centers are outside of the image
        if (bbxc<0 || bbxc>=template_width || bbyc<0 || bbyc>=template_height) {
          windows[ww].ignore = true;
        }
        //ignore gt bboxes smaller than minimum size
        if (bbw < min_gt_width || bbh < min_gt_height) {
          windows[ww].ignore = true;
        }     
        // for output gt boxes
        vector<Dtype> gt_box(8);
        gt_box[0] = item_id; 
        gt_box[1] = windows[ww].xmin; gt_box[2] = windows[ww].ymin; 
        gt_box[3] = windows[ww].xmax; gt_box[4] = windows[ww].ymax;
        gt_box[5] = windows[ww].label; 
        gt_box[6] = windows[ww].ignore;
        gt_box[7] = windows[ww].difficult;
        gt_boxes.push_back(gt_box);
      }

      // label data transfer
      for (int nn = 0; nn < label_blob_num_; nn++) {
        const int label_height = round(template_height/float(strides_[nn]));
        const int label_width = round(template_width/float(strides_[nn]));
        const int label_offset_x = round(src_offset_x/float(strides_[nn]));
        const int label_offset_y = round(src_offset_y/float(strides_[nn]));
        const int label_copy_width = round(copy_width/float(strides_[nn]));
        const int label_copy_height = round(copy_height/float(strides_[nn]));
        const int spatial_dim = label_height*label_width;
        CHECK_EQ(spatial_dim,label_spatial_dims[nn]);
        const Dtype radius_w = field_ws_[nn] / Dtype(2);
        const Dtype radius_h = field_hs_[nn] / Dtype(2);

        for (int h = 0; h < label_height; ++h) {
          for (int w = 0; w < label_width; ++w) {
            int top_index = (item_id*label_channel_*label_height+h)*label_width+w;
            if (w < label_offset_x || w >= label_offset_x+label_copy_width
                || h < label_offset_y || h >= label_offset_y+label_copy_height) {
              top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
              continue;
            }
            Dtype xx1, yy1, xx2, yy2;
            xx1 = (w+0.5)*strides_[nn]-radius_w; 
            xx2 = (w+0.5)*strides_[nn]+radius_w;
            yy1 = (h+0.5)*strides_[nn]-radius_h; 
            yy2 = (h+0.5)*strides_[nn]+radius_h;
            BBox anchor_bbox = InitBBox(xx1,yy1,xx2,yy2);
            
            // ignore bad anchor bboxes
            if (detection_data_param.has_anchor_ignore_margin_ratio()) {
              const float ignore_margin_ratio = detection_data_param.anchor_ignore_margin_ratio();
              const float width_margin = ignore_margin_ratio*radius_w;
              const float height_margin = ignore_margin_ratio*radius_h;
              if (xx1<src_offset_x-width_margin || xx2>=src_offset_x+copy_width+width_margin
                  || yy1<src_offset_y-height_margin || yy2>=src_offset_y+copy_height+height_margin) {
                top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
                continue;
              }
            } else if (detection_data_param.has_anchor_ignore_iou_thr()) {
              BBox image_bbox = InitBBox(src_offset_x,src_offset_y,
                      src_offset_x+copy_width,src_offset_y+copy_height);
              const float ignore_iou_thr = detection_data_param.anchor_ignore_iou_thr();
              float valid_iou = JaccardOverlap(anchor_bbox, image_bbox, "IOFU");
              if (valid_iou < ignore_iou_thr) {
                top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
                continue;
              }
            }
            
            // calclulate the iou of anchor bbox overlapping with ignored regions
            float sum_ignore_iou = 0;
            for (int ww = 0; ww < roni_windows.size(); ++ww) {
              float iou = JaccardOverlap(anchor_bbox, roni_windows[ww], "IOFU");
              sum_ignore_iou += iou;
            }

            int match_idx = -1; float max_iou = 0;
            for (int ww = 0; ww < windows.size(); ++ww) {
              Dtype iou = JaccardOverlap(windows[ww], anchor_bbox, "IOU");
              // find the best matched gt bbox for anchor bbox
              if (iou > max_iou) {
                match_idx = ww; max_iou = iou;
              }
              // find the best matched anchor bbox for gt bbox
              if (iou > max_gt_iou[ww]) {
                max_gt_iou[ww] = iou; max_gt_blobid[ww] = nn; 
              }
            }  
            if (max_iou > fg_threshold) {
              float x1 = windows[match_idx].xmin;
              float y1 = windows[match_idx].ymin;
              float x2 = windows[match_idx].xmax;
              float y2 = windows[match_idx].ymax;
              if (!windows[match_idx].ignore) {
                top_labels[nn][top_index] = windows[match_idx].label;
              } else {
                top_labels[nn][top_index] = 0;
              }
              top_labels[nn][top_index+spatial_dim] = (x1+x2)/Dtype(2.0);
              top_labels[nn][top_index+2*spatial_dim] = (y1+y2)/Dtype(2.0);
              top_labels[nn][top_index+3*spatial_dim] = x2-x1+1;
              top_labels[nn][top_index+4*spatial_dim] = y2-y1+1;
              top_labels[nn][top_index+5*spatial_dim] = max_iou;
              gt_match_counts[match_idx]++;
            } else if (sum_ignore_iou >= ignore_fg_threshold) {
              top_labels[nn][top_index+5*spatial_dim] = Dtype(1);
            } else {
              top_labels[nn][top_index+5*spatial_dim] = max_iou;
            }
          }
        }
      }

      //pick up those gt bboxes that are not matched yet
      if (label_blob_num_ > 0) {
        for (int ww = 0; ww < windows.size(); ++ww) {
          if (!windows[ww].ignore && gt_match_counts[ww] <= 0){ 
            float pickup_iou_thr = detection_data_param.pickup_iou_thr();
            if (max_gt_iou[ww] < pickup_iou_thr) {
              continue;
            }
            int miss_blobid = max_gt_blobid[ww];
            const int label_height = round(template_height/float(strides_[miss_blobid]));
            const int label_width = round(template_width/float(strides_[miss_blobid]));
            const int spatial_dim = label_height*label_width;

            const float x1 = windows[ww].xmin;
            const float y1 = windows[ww].ymin;
            const float x2 = windows[ww].xmax;
            const float y2 = windows[ww].ymax;
            Dtype bbxc = (x1+x2)/Dtype(2.0), bbyc = (y1+y2)/Dtype(2.0);
            int hc = floor(bbyc/strides_[miss_blobid]);
            hc = std::max(0,hc); hc = std::min(label_height-1,hc);
            int wc = floor(bbxc/strides_[miss_blobid]);
            wc = std::max(0,wc); wc = std::min(label_width-1,wc);
            const int miss_index = (item_id*label_channel_*label_height+hc)*label_width+wc;
            if (top_labels[miss_blobid][miss_index] > 0) {
              continue; 
            }
            top_labels[miss_blobid][miss_index] = windows[ww].label;
            top_labels[miss_blobid][miss_index+spatial_dim] = bbxc; 
            top_labels[miss_blobid][miss_index+2*spatial_dim] = bbyc;
            top_labels[miss_blobid][miss_index+3*spatial_dim] = x2-x1+1;
            top_labels[miss_blobid][miss_index+4*spatial_dim] = y2-y1+1;
            top_labels[miss_blobid][miss_index+5*spatial_dim] = max_gt_iou[ww];
          }
        } 
      }
    
      label_time += timer.MicroSeconds();
      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << list_id_;
      ss >> file_id;
      string root_dir = string("dump/");
      string outputstr = root_dir + file_id + string("_info.txt");
      std::ofstream inf(outputstr.c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << do_mirror << std::endl;
      for (int ww = 0; ww < windows.size(); ++ww) {
        inf << windows[ww].label <<", "
            << int(windows[ww].ignore) <<", "
            << windows[ww].xmin <<", "
            << windows[ww].ymin <<", "
            << windows[ww].xmax <<", "
            << windows[ww].ymax << std::endl;
      }
      for (int ww = 0; ww < roni_windows.size(); ++ww) {
        inf << 100 << ", "<< 100 << ", "  
            << roni_windows[ww].xmin <<", "
            << roni_windows[ww].ymin <<", "
            << roni_windows[ww].xmax <<", "
            << roni_windows[ww].ymax << std::endl;
      }
      inf.close();
      
      std::ofstream top_data_file((root_dir + file_id + string("_data.txt")).c_str(),
          std::ofstream::out);
      for (int c = 0; c < channels; ++c) {
        for (int w = 0; w < template_width; ++w) {
          for (int h = 0; h < template_height; ++h) {
            top_data_file << top_data[((item_id * channels + c) * template_height + h)
                          * template_width + w]<<std::endl;
          }
        }
      }
      top_data_file.close();
      
      for (int nn = 0; nn < label_blob_num_; nn++) {
        const int label_height = round(template_height/float(strides_[nn]));
        const int label_width = round(template_width/float(strides_[nn]));
        string label_id; std::stringstream sss;
        sss << nn; sss >> label_id;
        std::ofstream top_label_file((root_dir + file_id + string("_")+label_id
                +string("_label.txt")).c_str(),std::ofstream::out);
        for (int k = 0; k < label_channel_; k++) {
          for (int w = 0; w < label_width; ++w) {
            for (int h = 0; h < label_height; ++h) {
              top_label_file << top_labels[nn][((item_id * label_channel_ + k) * label_height + h) 
                             * label_width + w]<<std::endl;
            }
          }
        }
        top_label_file.close();
      }

      #endif

      list_id_++;
      //TODO: reshuffle in the middle of a batch could be a bug for mix shuffling.
      if (list_id_ >= image_database_size) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        list_id_ = 0;
        if (detection_data_param.has_shuffle()) {
          if (detection_data_param.shuffle() == "random_aspect") {
            LOG(INFO) << "Random aspect shuffling data";
            ShuffleAspectGroupList();
          } else if (detection_data_param.shuffle() == "random_mix_aspect") {
            LOG(INFO) << "Random mixture shuffling data";
            ShuffleMixAspectGroupList();
          } else if (detection_data_param.shuffle() == "random") {
            LOG(INFO) << "Random shuffling data";
            ShuffleList();
          } else {
            CHECK(false) << "Unknown shuffling Function!";
          }
        }
      }
  }
  
  // output gt boxes [img_id, xmin, ymin, xmax, ymax, label, ignored, difficult]
  if (output_gt_boxes_) {
    int num_gt_boxes = gt_boxes.size();
    const int gt_dim = 8;
    // for special case when there is no gt
    if (num_gt_boxes <= 0) {
      batch->labels_[label_blob_num_]->Reshape(1, gt_dim, 1, 1);
      Dtype* gt_boxes_data = batch->labels_[label_blob_num_]->mutable_cpu_data();
      caffe_set<Dtype>(gt_dim, -1, gt_boxes_data);
    } else {
      batch->labels_[label_blob_num_]->Reshape(num_gt_boxes, gt_dim, 1, 1);
      Dtype* gt_boxes_data = batch->labels_[label_blob_num_]->mutable_cpu_data();
      for (int i = 0; i < num_gt_boxes; i++) {
        for (int j = 0; j < gt_dim; j++) {
          gt_boxes_data[i*gt_dim+j] = gt_boxes[i][j];
        }
      }
    }
  }
  
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  DLOG(INFO) << "Label time: " << label_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DetectionDataLayer);
REGISTER_LAYER_CLASS(DetectionData);

}  // namespace caffe
#endif  // USE_OPENCV
