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
#include "caffe/layers/eval_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"

// caffe.proto > LayerParameter > EvalDataParameter

namespace caffe {

template <typename Dtype>
EvalDataLayer<Dtype>::~EvalDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void EvalDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  const EvalDataParameter& eval_data_param = 
          this->layer_param_.eval_data_param();
  
  LOG(INFO) << "Window data layer:" << std::endl
      << "  batch size: "
      << eval_data_param.batch_size() << std::endl
      << "  root_folder: "
      << eval_data_param.root_folder();

  string root_folder = eval_data_param.root_folder();
  std::ifstream infile(eval_data_param.source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << eval_data_param.source() << std::endl;

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

  LOG(INFO) << "Number of images: " << image_index+1;
  CHECK_EQ(windows_.size(),image_database_.size());
  CHECK_EQ(windows_.size(),roni_windows_.size());
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }
   
  list_id_ = 0;
  
  // image reshape
  const int batch_size = eval_data_param.batch_size();
  if (!(eval_data_param.has_resize_width() && 
          eval_data_param.has_resize_height())) {
    // make sure all images in the same batch have same dimensions
    CHECK_EQ(batch_size,1);
  }
  int template_height, template_width;
  template_width = img_width; template_height = img_height;
  if (eval_data_param.has_resize_width()
      && eval_data_param.has_resize_height()) {
    template_width = eval_data_param.resize_width();
    template_height = eval_data_param.resize_height();
  } else if (eval_data_param.has_short_size()
      && eval_data_param.has_long_size()) {
    template_width = eval_data_param.short_size();
    template_height = eval_data_param.long_size();
  }
  top[0]->Reshape(batch_size, channels, template_height, template_width);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(batch_size, channels, template_height, 
            template_width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
  
  // setup for output gt boxes
  const int gt_dim = 8;
  top[1]->Reshape(1, gt_dim, 1, 1);    
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
    label_blob_pointer->Reshape(1, gt_dim, 1, 1);
    this->prefetch_[i]->labels_.push_back(label_blob_pointer);
  }
  LOG(INFO) << "output gt boxes size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();

  // setup for output gt boxes
  output_image_size_ = top.size()>=3;
  if (output_image_size_) {
    top[2]->Reshape(batch_size, 2, 1, 1);    
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(batch_size, 2, 1, 1);
      this->prefetch_[i]->labels_.push_back(label_blob_pointer);
    }
    LOG(INFO) << "output image dims: " << top[2]->num() << ","
        << top[2]->channels() << "," << top[2]->height() << "," << top[2]->width();
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

// Thread fetching the data
template <typename Dtype>
void EvalDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer, timer;
  batch_timer.Start();
  double read_time = 0, trans_time = 0;
  Dtype* top_data = NULL;

  const EvalDataParameter& eval_data_param = 
          this->layer_param_.eval_data_param();
  const Dtype scale = this->transform_param_.scale();
  const int batch_size = eval_data_param.batch_size();
   
  // gt boxes
  vector<vector<Dtype> > gt_boxes;
  int image_database_size = image_database_.size();
  // sample from bg set then fg set
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      // sample a window
      timer.Start();
      CHECK_GT(image_database_size, list_id_);
      //const unsigned int rand_index = PrefetchRand();
      const unsigned int image_index = image_list_[list_id_];
      vector<BBox> windows = windows_[image_index];
      vector<BBox> roni_windows = roni_windows_[image_index];
      
      // load the image containing the window
      pair<std::string, vector<int> > image = image_database_[image_index];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
      
      int img_height = cv_img.rows, img_width = cv_img.cols;
      read_time += timer.MicroSeconds();
      timer.Start();
             
      // resize image if needed
      int resize_width = img_width, resize_height = img_height;
      if (eval_data_param.has_resize_width()
          && eval_data_param.has_resize_height()) {
        resize_width = eval_data_param.resize_width();
        resize_height = eval_data_param.resize_height();  
      } else if (eval_data_param.has_short_size() &&
          eval_data_param.has_long_size()) {
        int short_size = eval_data_param.short_size();
        int long_size = eval_data_param.long_size();
        int img_short_size = std::min(img_width,img_height);
        float resize_scale = float(short_size)/float(img_short_size);
        resize_width = round(img_width*resize_scale);
        resize_height = round(img_height*resize_scale);
        resize_width = std::min(resize_width,long_size);
        resize_height = std::min(resize_height,long_size);
      }
      resize_width = round(resize_width/32.0)*32;
      resize_height = round(resize_height/32.0)*32;
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

      // blob reshape before copy
      if (item_id==0) {
        // data
        int channels = batch->data_.channels();
        batch->data_.Reshape(batch_size, channels, resize_height, resize_width);
        batch->labels_[1]->Reshape(batch_size, 2, 1, 1);
        top_data = batch->data_.mutable_cpu_data();
        // zero out data batch
        caffe_set(batch->data_.count(), Dtype(0), top_data);
      }
      
      // assign image sizes (height, width) if needed
      if (output_image_size_) {
        Dtype* image_size_data = batch->labels_[1]->mutable_cpu_data();
        image_size_data[item_id*2] = resize_height;
        image_size_data[item_id*2+1] = resize_width;
      }

      // copy the original image into top_data
      const int channels = cv_img.channels();     
      for (int h = 0; h < resize_height; ++h) {
        const uchar* ptr = cv_img.ptr<uchar>(h);
        for (int w = 0; w < resize_width; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id*channels+c)*resize_height+h)*resize_width+w;
            int img_index = w * channels + c;
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
      
      // get window label
      for (int ww = 0; ww < windows.size(); ++ww) {
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
     
      list_id_++;
      if (list_id_ >= image_database_size) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        list_id_ = 0;
      }
  }
  
  // output gt boxes [img_id, xmin, ymin, xmax, ymax, label, ignored, difficult]
  int num_gt_boxes = gt_boxes.size();
  const int gt_dim = 8;
  if (num_gt_boxes == 0) {
    // for special case when there is no gt
    batch->labels_[0]->Reshape(1, gt_dim, 1, 1);
    Dtype* gt_boxes_data = batch->labels_[0]->mutable_cpu_data();
    caffe_set<Dtype>(gt_dim, -1, gt_boxes_data);
  } else {
    batch->labels_[0]->Reshape(num_gt_boxes, gt_dim, 1, 1);
    Dtype* gt_boxes_data = batch->labels_[0]->mutable_cpu_data();
    for (int i = 0; i < num_gt_boxes; i++) {
      for (int j = 0; j < gt_dim; j++) {
        gt_boxes_data[i*gt_dim+j] = gt_boxes[i][j];
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(EvalDataLayer);
REGISTER_LAYER_CLASS(EvalData);

}  // namespace caffe
#endif  // USE_OPENCV
