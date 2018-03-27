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

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/util/bbox_util.hpp"
#include "caffe/layers/frcnn_output_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void FrcnnOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  bbox_cls_aware_ = this->layer_param_.bbox_reg_param().cls_aware();
   
  // bbox mean and std
  bbox_mean_.Reshape(4,1,1,1); bbox_std_.Reshape(4,1,1,1);
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    int num_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_means,4); CHECK_EQ(num_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_mean_.mutable_cpu_data()[i] = this->layer_param_.bbox_reg_param().bbox_mean(i);
      bbox_std_.mutable_cpu_data()[i] = this->layer_param_.bbox_reg_param().bbox_std(i);
      CHECK_GT(bbox_std_.mutable_cpu_data()[i],0);
    }
  } else {
    caffe_set(bbox_mean_.count(), Dtype(0), bbox_mean_.mutable_cpu_data());
    caffe_set(bbox_std_.count(), Dtype(1), bbox_std_.mutable_cpu_data());
  }
   
  // detection saving
  const SaveOutputParameter& save_output_param =
      this->layer_param_.frcnn_output_param().save_output_param();
  output_directory_ = save_output_param.output_directory();
  if (!output_directory_.empty()) {
    if (boost::filesystem::is_directory(output_directory_)) {
      boost::filesystem::remove_all(output_directory_);
    }
    if (!boost::filesystem::create_directories(output_directory_)) {
        LOG(WARNING) << "Failed to create directory: " << output_directory_;
    }
  }
  output_name_prefix_ = save_output_param.output_name_prefix();
  need_save_ = output_directory_ == "" ? false : true;
  output_format_ = save_output_param.output_format();
  if (save_output_param.has_label_map_file()) {
    string label_map_file = save_output_param.label_map_file();
    if (label_map_file.empty()) {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
      need_save_ = false;
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  } else {
    need_save_ = false;
  }
  if (save_output_param.has_name_size_file()) {
    string name_size_file = save_output_param.name_size_file();
    if (name_size_file.empty()) {
      // Ignore saving if there is no name_size_file provided.
      LOG(WARNING) << "Provide name_size_file if output results to files.";
      need_save_ = false;
    } else {
      std::ifstream infile(name_size_file.c_str());
      CHECK(infile.good())
          << "Failed to open name size file: " << name_size_file;
      // The file is in the following format:
      //    name height width
      //    ...
      string name;
      int height, width;
      while (infile >> name >> height >> width) {
        names_.push_back(name);
        sizes_.push_back(std::make_pair(height, width));
      }
      infile.close();
      if (save_output_param.has_num_test_image()) {
        num_test_image_ = save_output_param.num_test_image();
      } else {
        num_test_image_ = names_.size();
      }
      CHECK_LE(num_test_image_, names_.size());
    }
  } else {
    need_save_ = false;
  }
  name_count_ = 0;
}

template <typename Dtype>
void FrcnnOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (need_save_) {
    CHECK_LE(name_count_, names_.size());
    if (name_count_ % num_test_image_ == 0) {
      // Clean all outputs.
      if (output_format_ == "VOC") {
        boost::filesystem::path output_directory(output_directory_);
        for (map<int, string>::iterator it = label_to_name_.begin();
             it != label_to_name_.end(); ++it) {
          if (it->first == 0) {
            continue;
          }
          std::ofstream outfile;
          boost::filesystem::path file(
              output_name_prefix_ + it->second + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          outfile.open(out_file.string().c_str(), std::ofstream::out);
        }
      }
    }
  }
  
  // bottom: prob_blob, bbox_blob, prior_blob, image_sizes
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->num(),bottom[2]->num());
  CHECK((bottom[2]->channels()==5) || (bottom[2]->channels()==6));
  if (bbox_cls_aware_) {
    CHECK_EQ(4*bottom[0]->channels(),bottom[1]->channels());
  } else {
    CHECK_EQ(bottom[1]->channels(),8);
  }
  CHECK_EQ(bottom[3]->num(),1);
  CHECK_EQ(bottom[3]->channels(),2);
      
  //dummy reshape 
  //[image_id, xmin, ymin, xmax, ymax, label, confidence]
  top[0]->Reshape(1, 7, 1, 1);
}

template <typename Dtype>
void FrcnnOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // bottom: prob_blob, bbox_blob, prior_blob, image_sizes
  const int num = 1;
  const int instance_num = bottom[0]->num();
  const int cls_num = bottom[0]->channels();
  const int bbox_dim = bottom[1]->channels();
  const int prior_dim = bottom[2]->channels();
  // input size (height, width)
  const Dtype * size_data = bottom[3]->cpu_data();
  
  FrcnnOutputParameter frcnn_output_param = this->layer_param_.frcnn_output_param();
  const float conf_thr = frcnn_output_param.conf_thr();
  const float iou_thr = frcnn_output_param.iou_thr();
  const string nms_type = frcnn_output_param.nms_type();
  const int max_det_num = frcnn_output_param.max_det_num(); 
  
  // prob values.
  const Dtype* prob_data = bottom[0]->cpu_data();
  
  //decode prior box [img_id x1 y1 x2 y2 (score)]
  const Dtype* prior_data = bottom[2]->cpu_data(); 
  vector<BBox> prior_bboxes;
  for (int i = 0; i < instance_num; i++) {
    CHECK_EQ(prior_data[i*prior_dim],0); // batch size == 1!
    BBox bbox;
    bbox.xmin = prior_data[i*prior_dim + 1];
    bbox.ymin = prior_data[i*prior_dim + 2];
    bbox.xmax = prior_data[i*prior_dim + 3];
    bbox.ymax = prior_data[i*prior_dim + 4];
    CHECK_GE(bbox.xmax,bbox.xmin);
    CHECK_GE(bbox.ymax,bbox.ymin);
    if (prior_dim == 6) {
      bbox.score = prior_data[i*prior_dim + 5];
    }
    bbox.size = BBoxSize(bbox);
    prior_bboxes.push_back(bbox);
  }
   
  // decode bbox predictions
  const Dtype* bbox_data = bottom[1]->cpu_data();
  Dtype* bbox_pred_data = bottom[1]->mutable_cpu_diff();
  
  DecodeBBoxesWithPrior(bbox_data,prior_bboxes,bbox_dim,bbox_mean_.cpu_data(),
          bbox_std_.cpu_data(),bbox_pred_data);
         
  int num_all_boxes = 0;
  vector<vector<vector<BBox> > > all_bboxes(num);
  for (int i = 0; i < num; i++) { 
    vector<vector<BBox> > cls_bboxes(cls_num-1);
    int num_det_img = 0;
    for (int c = 1; c < cls_num; c++) {
      vector<BBox> bboxes;
      int bb_count = 0;
      for (int id = 0; id < instance_num; id++) {
        Dtype cls_score = prob_data[id*cls_num+c];
        if (cls_score >= conf_thr) {
          BBox bbox; 
          int base_index;
          if (bbox_cls_aware_) {
            base_index = id*bbox_dim+c*4;
          } else {
            base_index = id*bbox_dim+4;
          }
          bbox.xmin = bbox_pred_data[base_index]; 
          bbox.ymin = bbox_pred_data[base_index+1]; 
          bbox.xmax = bbox_pred_data[base_index+2]; 
          bbox.ymax = bbox_pred_data[base_index+3];
          bbox.score = cls_score;
          ClipBBox(bbox, size_data[i*2+1], size_data[i*2]);
          bboxes.push_back(bbox);
          bb_count++;
        }
      }
      if (bb_count<=0) continue;
      //ranking decreasingly
      std::sort(bboxes.begin(),bboxes.end(),SortBBoxDescend);
      vector<BBox> new_bboxes;
      //NMS, make sure the scores are already ranked
      new_bboxes = ApplyNMS(bboxes, iou_thr, true, nms_type);
      num_det_img += new_bboxes.size();
      cls_bboxes[c-1] = new_bboxes;
    }
    if (max_det_num > 0 && num_det_img > max_det_num) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (int c = 1; c < cls_num; c++) {
        for (int j = 0; j < cls_bboxes[c-1].size(); j++) {
          score_index_pairs.push_back(std::make_pair(
                  cls_bboxes[c-1][j].score, std::make_pair(c, j)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(max_det_num);
      // Store the new indices.
      vector<vector<BBox> > new_cls_bboxes(cls_num-1);
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        BBox bbox = cls_bboxes[label-1][idx];
        new_cls_bboxes[label-1].push_back(bbox);
      }
      all_bboxes[i] = new_cls_bboxes;
      num_all_boxes += max_det_num;
    } else {
      all_bboxes[i] = cls_bboxes;
      num_all_boxes += num_det_img;
    } 
  }
  if (num_all_boxes <= 0) {
    LOG(INFO) << "Couldn't find any detections";
  }
  // output detections [image_id, xmin, ymin, xmax, ymax, label, confidence]
  const int det_dim = 7;
  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  // add a fake detection to each image
  top[0]->Reshape(num_all_boxes+num, det_dim, 1, 1);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(-1), top_data);

  for (int i = 0; i < num; ++i) {
    // add a fake detection to each image
    top_data[count * det_dim] = i; ++count;
    const vector<vector<BBox> > cls_bboxes = all_bboxes[i];
    for (int c = 1; c < cls_num; c++) {
      const vector<BBox> bboxes = cls_bboxes[c-1];
      for (int j = 0; j < bboxes.size(); j++) {
        top_data[count * det_dim] = i;
        top_data[count * det_dim + 1] = bboxes[j].xmin;
        top_data[count * det_dim + 2] = bboxes[j].ymin;
        top_data[count * det_dim + 3] = bboxes[j].xmax;
        top_data[count * det_dim + 4] = bboxes[j].ymax;
        top_data[count * det_dim + 5] = c;
        top_data[count * det_dim + 6] = bboxes[j].score;
        if (need_save_) {
          BBox out_bbox = bboxes[j];
          // scale ground truth and detections to actual values
          const float h_ratio = sizes_[name_count_].first / size_data[i*2];
          const float w_ratio = sizes_[name_count_].second / size_data[i*2+1];
          AffineBBox(out_bbox,w_ratio,h_ratio,0,0);
          
          float score = out_bbox.score;
          float xmin = out_bbox.xmin;
          float ymin = out_bbox.ymin;
          float xmax = out_bbox.xmax;
          float ymax = out_bbox.ymax;
          ptree pt_xmin, pt_ymin, pt_width, pt_height;
          pt_xmin.put<float>("", round(xmin * 100) / 100.);
          pt_ymin.put<float>("", round(ymin * 100) / 100.);
          pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
          pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

          ptree cur_bbox;
          cur_bbox.push_back(std::make_pair("", pt_xmin));
          cur_bbox.push_back(std::make_pair("", pt_ymin));
          cur_bbox.push_back(std::make_pair("", pt_width));
          cur_bbox.push_back(std::make_pair("", pt_height));

          ptree cur_det;
          cur_det.put("image_id", names_[name_count_]);
          if (output_format_ == "ILSVRC") {
            cur_det.put<int>("category_id", c);
          } else {
            cur_det.put("category_id", label_to_name_[c].c_str());
          }
          cur_det.add_child("bbox", cur_bbox);
          cur_det.put<float>("score", score);

          detections_.push_back(std::make_pair("", cur_det));
        }
        ++count;
      }
    }
    if (need_save_) {
      ++name_count_;
      if (name_count_ % num_test_image_ == 0) {
        if (output_format_ == "VOC") {
          map<string, std::ofstream*> outfiles;
          for (int c = 1; c < cls_num; ++c) {
            string label_name = label_to_name_[c];
            boost::filesystem::path file(
                output_name_prefix_ + label_name + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
                std::ofstream::out);
          }
          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            string label_name = pt.get<string>("category_id");
            if (outfiles.find(label_name) == outfiles.end()) {
              std::cout << "Cannot find " << label_name << std::endl;
              continue;
            }
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            *(outfiles[label_name]) << image_name;
            *(outfiles[label_name]) << " " << score;
            *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
            *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
            *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
            *(outfiles[label_name]) << std::endl;
          }
          for (int c = 1; c < cls_num; ++c) {
            string label_name = label_to_name_[c];
            outfiles[label_name]->flush();
            outfiles[label_name]->close();
            delete outfiles[label_name];
          }
        }
        name_count_ = 0;
        detections_.clear();
      }
    }
  }
  CHECK_EQ(top[0]->num(),count);
}

INSTANTIATE_CLASS(FrcnnOutputLayer);
REGISTER_LAYER_CLASS(FrcnnOutput);

}  // namespace caffe
